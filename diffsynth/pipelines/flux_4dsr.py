from typing import Union, Optional
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat
from ..utils import ModelConfig, BasePipeline, PipelineUnit, PipelineUnitRunner
from ..models import (
    FluxDiT,
    ModelManager,
    FluxVAEDecoder,
    FluxVAEEncoder,
    SD3TextEncoder1,
    FluxTextEncoder2,
    load_state_dict,
)
from ..prompters import FluxPrompter
from ..schedulers import FlowMatchScheduler
from ..models.tiler import FastTileWorker
from ..lora.flux_lora import FluxLoRAFuser, FluxLoRALoader, FluxLoraPatcher
from ..models.flux_dit import RMSNorm
from ..vram_management import (
    AutoWrappedLinear,
    AutoWrappedModule,
    enable_vram_management,
    gradient_checkpoint_forward,
)
from ..models.flux_ipadapter import FluxIpAdapter
from ..models.flux_controlnet import FluxControlNet

from easyvolcap.utils.console_utils import log
from FDL_pytorch import FDL_loss

@dataclass
class ControlNetInput:
    controlnet_id: int = 0
    scale: float = 1.0
    start: float = 1.0
    end: float = 0.0
    image: Image.Image = None
    inpaint_mask: Image.Image = None
    processor_id: str = None



class MultiControlNet(torch.nn.Module):
    def __init__(self, models: list[FluxControlNet]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def process_single_controlnet(self, controlnet_input: ControlNetInput, conditioning: torch.Tensor, **kwargs):
        model = self.models[controlnet_input.controlnet_id]
        res_stack, single_res_stack = model(
            controlnet_conditioning=conditioning,
            processor_id=controlnet_input.processor_id,
            **kwargs
        )
        res_stack = [res * controlnet_input.scale for res in res_stack]
        single_res_stack = [res * controlnet_input.scale for res in single_res_stack]
        return res_stack, single_res_stack

    def forward(self, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, **kwargs):
        res_stack, single_res_stack = None, None
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start or progress < controlnet_input.end:
                continue
            res_stack_, single_res_stack_ = self.process_single_controlnet(controlnet_input, conditioning, **kwargs)
            if res_stack is None:
                res_stack = res_stack_
                single_res_stack = single_res_stack_
            else:
                res_stack = [i + j for i, j in zip(res_stack, res_stack_)]
                single_res_stack = [i + j for i, j in zip(single_res_stack, single_res_stack_)]
        return res_stack, single_res_stack



class Flux4DSRPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler()
        self.prompter = FluxPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.controlnet: MultiControlNet = None
        self.ipadapter: FluxIpAdapter = None
        self.ipadapter_image_encoder = None
        self.lora_patcher: FluxLoraPatcher = None
        self.unit_runner = PipelineUnitRunner()
        self.in_iteration_models = ("dit", "controlnet", "lora_patcher")
        self.units = [
            FluxImageUnit_ShapeChecker(),
            FluxImageUnit_NoiseInitializer(),
            FluxImageUnit_PromptEmbedder(),
            FluxImageUnit_InputImageEmbedder(),
            FluxImageUnit_ImageIDs(),
            FluxImageUnit_EmbeddedGuidanceEmbedder(),
            FluxImageUnit_Kontext(),
            FluxImageUnit_ControlNet(),
            FluxImageUnit_IPAdapter(),
            FluxImageUnit_TeaCache(),
        ]
        self.model_fn = model_fn_flux_image


    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        loader = FluxLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = loader.convert_state_dict(lora)
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader.load(module, lora, alpha=alpha)


    def load_loras(
        self,
        module: torch.nn.Module,
        lora_configs: list[Union[ModelConfig, str]],
        alpha=1,
        hotload=False,
        extra_fused_lora=False,
    ):
        for lora_config in lora_configs:
            self.load_lora(module, lora_config, hotload=hotload, alpha=alpha)
        if extra_fused_lora:
            lora_fuser = FluxLoRAFuser(device="cuda", torch_dtype=torch.bfloat16)
            fused_lora = lora_fuser(lora_configs)
            self.load_lora(module, state_dict=fused_lora, hotload=hotload, alpha=alpha)


    def clear_lora(self):
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear):
                if hasattr(module, "lora_A_weights"):
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()


    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)

        # TODO: add FDL loss
        if inputs["use_fdl_loss"]:
            fdl_loss = FDL_loss(num_proj=128, model="VGG")
            loss += fdl_loss(noise_pred.float(), training_target.float()) * inputs["fdl_loss_weights"]
        return loss


    def _enable_vram_management_with_default_config(self, model, vram_limit):
        if model is not None:
            dtype = next(iter(model.parameters())).dtype
            enable_vram_management(
                model,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.GroupNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )


    def enable_lora_magic(self):
        if self.dit is not None:
            if not (hasattr(self.dit, "vram_management_enabled") and self.dit.vram_management_enabled):
                dtype = next(iter(self.dit.parameters())).dtype
                enable_vram_management(
                    self.dit,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device=self.device,
                        onload_dtype=dtype,
                        onload_device=self.device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=None,
                )
        if self.lora_patcher is not None:
            for name, module in self.dit.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    merger_name = name.replace(".", "___")
                    if merger_name in self.lora_patcher.model_dict:
                        module.lora_merger = self.lora_patcher.model_dict[merger_name]


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer

        # Default config
        default_vram_management_models = ["text_encoder_1", "vae_decoder", "vae_encoder", "controlnet", "ipadapter", "lora_patcher"]
        for model_name in default_vram_management_models:
            self._enable_vram_management_with_default_config(getattr(self, model_name), vram_limit)

        # Special config
        if self.text_encoder_2 is not None:
            from transformers.models.t5.modeling_t5 import (
                T5LayerNorm,
                T5DenseActDense,
                T5DenseGatedActDense,
            )
            dtype = next(iter(self.text_encoder_2.parameters())).dtype
            enable_vram_management(
                self.text_encoder_2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                    T5DenseActDense: AutoWrappedModule,
                    T5DenseGatedActDense: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.ipadapter_image_encoder is not None:
            from transformers.models.siglip.modeling_siglip import (
                SiglipEncoder,
                SiglipVisionEmbeddings,
                SiglipMultiheadAttentionPoolingHead,
            )
            dtype = next(iter(self.ipadapter_image_encoder.parameters())).dtype
            enable_vram_management(
                self.ipadapter_image_encoder,
                module_map = {
                    SiglipVisionEmbeddings: AutoWrappedModule,
                    SiglipEncoder: AutoWrappedModule,
                    SiglipMultiheadAttentionPoolingHead: AutoWrappedModule,
                    torch.nn.MultiheadAttention: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
    ):
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        # Initialize pipeline
        pipe = Flux4DSRPipeline(device=device, torch_dtype=torch_dtype)
        pipe.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        pipe.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        pipe.dit = model_manager.fetch_model("flux_dit")
        pipe.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        pipe.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        pipe.prompter.fetch_models(pipe.text_encoder_1, pipe.text_encoder_2)
        pipe.ipadapter = model_manager.fetch_model("flux_ipadapter")
        pipe.ipadapter_image_encoder = model_manager.fetch_model("siglip_vision_model")
        pipe.lora_patcher = model_manager.fetch_model("flux_lora_patcher")

        # ControlNet
        controlnets = []
        for model_name, model in zip(model_manager.model_name, model_manager.model):
            if model_name == "flux_controlnet":
                controlnets.append(model)
        if len(controlnets) > 0:
            pipe.controlnet = MultiControlNet(controlnets)

        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 3.5,
        t5_sequence_length: int = 512,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Scheduler
        sigma_shift: float = None,
        # Steps
        num_inference_steps: int = 30,
        # local prompts
        multidiffusion_prompts=(),
        multidiffusion_masks=(),
        multidiffusion_scales=(),
        # Kontext
        kontext_images: Union[list[Image.Image], Image.Image] = None,
        # Kontext reference image position offsets
        kontext_ref_offsets: list[int] = [1, 0, 0],
        # ControlNet
        controlnet_inputs: list[ControlNetInput] = None,
        # IP-Adapter
        ipadapter_images: Union[list[Image.Image], Image.Image] = None,
        ipadapter_scale: float = 1.0,
        # TeaCache
        tea_cache_l1_thresh: float = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance, "t5_sequence_length": t5_sequence_length,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "sigma_shift": sigma_shift, "num_inference_steps": num_inference_steps,
            "multidiffusion_prompts": multidiffusion_prompts, "multidiffusion_masks": multidiffusion_masks, "multidiffusion_scales": multidiffusion_scales,
            "kontext_images": kontext_images,
            "kontext_ref_offsets": kontext_ref_offsets,
            "controlnet_inputs": controlnet_inputs,
            "ipadapter_images": ipadapter_images, "ipadapter_scale": ipadapter_scale,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "progress_bar_cmd": progress_bar_cmd,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Decode
        self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image



class FluxImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"))

    def process(self, pipe: Flux4DSRPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class FluxImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"))

    def process(self, pipe: Flux4DSRPipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device)
        return {"noise": noise}



class FluxImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: Flux4DSRPipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae_encoder'])
        input_latents = []
        for img in input_image:
            image = pipe.preprocess_image(img).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latent = pipe.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            input_latents.append(input_latent)

        # TODO: concat the input latents in batch dim, from (B, C, H, W) -> (NxB, C, H, W)
        input_latents = torch.concat(input_latents, dim=0)
        noise = repeat(noise, '1 ... -> b ...', b=input_latents.shape[0])
        log(f"Input latents shape: {input_latents.shape}")
        log(f"Noise shape: {noise.shape}")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": None}



class FluxImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            input_params=("t5_sequence_length",),
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )

    def process(self, pipe: Flux4DSRPipeline, prompt, t5_sequence_length, positive) -> dict:
        if pipe.text_encoder_1 is not None and pipe.text_encoder_2 is not None:
            prompt_emb, pooled_prompt_emb, text_ids = pipe.prompter.encode_prompt(
                prompt, device=pipe.device, positive=positive, t5_sequence_length=t5_sequence_length
            )
            return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
        else:
            return {}


class FluxImageUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",))

    def process(self, pipe: Flux4DSRPipeline, latents):
        latent_image_ids = pipe.dit.prepare_image_ids(latents)
        return {"image_ids": latent_image_ids}



class FluxImageUnit_EmbeddedGuidanceEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("embedded_guidance", "latents"))

    def process(self, pipe: Flux4DSRPipeline, embedded_guidance, latents):
        guidance = torch.Tensor([embedded_guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"guidance": guidance}



class FluxImageUnit_Kontext(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("kontext_images", "tiled", "tile_size", "tile_stride", "kontext_ref_offsets"))

    def process(self, pipe: Flux4DSRPipeline, kontext_images, tiled, tile_size, tile_stride, kontext_ref_offsets):
        if kontext_images is None:
            return {}
        if not isinstance(kontext_images, list):
            kontext_images = [kontext_images]

        kontext_latents = []
        kontext_image_ids = []
        for kontext_image in kontext_images:
            kontext_image = pipe.preprocess_image(kontext_image)
            kontext_latent = pipe.vae_encoder(kontext_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            image_ids = pipe.dit.prepare_image_ids(kontext_latent)

            # image_ids[..., 0] = 1
            # add reference offsets to image_ids
            assert len(kontext_ref_offsets) == 3, "Kontext model uses 3 dim position offsets."
            for i in range(len(kontext_ref_offsets)):
                image_ids[..., i] += kontext_ref_offsets[i]
            kontext_image_ids.append(image_ids)
            kontext_latent = pipe.dit.patchify(kontext_latent)
            kontext_latents.append(kontext_latent)

        if len(kontext_images) > 1:
            # TODO: multi-inputs concat in batch dim
            kontext_latents = torch.concat(kontext_latents, dim=0)
            kontext_image_ids = torch.concat(kontext_image_ids, dim=0)
        else:
            # original concat in sequence dim
            kontext_latents = torch.concat(kontext_latents, dim=1)
            kontext_image_ids = torch.concat(kontext_image_ids, dim=-2)

        # log(f"Kontext latents shape: {kontext_latents.shape}")
        # log(f"Kontext image IDs shape: {kontext_image_ids.shape}")

        return {"kontext_latents": kontext_latents, "kontext_image_ids": kontext_image_ids}



class FluxImageUnit_ControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae_encoder",)
        )

    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: Flux4DSRPipeline, controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(['vae_encoder'])
        conditionings = []
        for controlnet_input in controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
        return {"controlnet_conditionings": conditionings}



class FluxImageUnit_IPAdapter(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("ipadapter_image_encoder", "ipadapter")
        )

    def process(self, pipe: Flux4DSRPipeline, inputs_shared, inputs_posi, inputs_nega):
        ipadapter_images, ipadapter_scale = inputs_shared.get("ipadapter_images", None), inputs_shared.get("ipadapter_scale", 1.0)
        if ipadapter_images is None:
            return inputs_shared, inputs_posi, inputs_nega
        if not isinstance(ipadapter_images, list):
            ipadapter_images = [ipadapter_images]

        pipe.load_models_to_device(self.onload_model_names)
        images = [image.convert("RGB").resize((384, 384), resample=3) for image in ipadapter_images]
        images = [pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype) for image in images]
        ipadapter_images = torch.cat(images, dim=0)
        ipadapter_image_encoding = pipe.ipadapter_image_encoder(ipadapter_images).pooler_output

        inputs_posi.update({"ipadapter_kwargs_list": pipe.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)})
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update({"ipadapter_kwargs_list": pipe.ipadapter(torch.zeros_like(ipadapter_image_encoding))})
        return inputs_shared, inputs_posi, inputs_nega


class FluxImageUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("num_inference_steps","tea_cache_l1_thresh"))

    def process(self, pipe: Flux4DSRPipeline, num_inference_steps, tea_cache_l1_thresh):
        if tea_cache_l1_thresh is None:
            return {}
        else:
            return {"tea_cache": TeaCache(num_inference_steps=num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh)}



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: FluxDiT, hidden_states, conditioning):
        inp = hidden_states.clone()
        temb_ = conditioning.clone()
        modulated_inp, _, _, _, _ = dit.blocks[0].norm1_a(inp, emb=temb_)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = hidden_states.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def model_fn_flux_image(
    dit: FluxDiT,
    controlnet=None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    kontext_latents=None,
    kontext_image_ids=None,
    controlnet_inputs=None,
    controlnet_conditionings=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    ipadapter_kwargs_list={},
    tea_cache: TeaCache = None,
    progress_id=0,
    num_inference_steps=1,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_conditionings = [f[:, :, hl: hr, wl: wr] for f in controlnet_conditionings] if controlnet_conditionings is not None else None
            return model_fn_flux_image(
                dit=dit,
                controlnet=controlnet,
                latents=latents[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_inputs=controlnet_inputs,
                controlnet_conditionings=tiled_controlnet_conditionings,
                tiled=False,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            latents,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=latents.device,
            tile_dtype=latents.dtype
        )

    hidden_states = latents

    # ControlNet
    if controlnet is not None and controlnet_conditionings is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "controlnet_inputs": controlnet_inputs,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "progress_id": progress_id,
            "num_inference_steps": num_inference_steps,
        }
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_conditionings, **controlnet_extra_kwargs
        )

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)

    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states) # (B, S, C), S is the sequence length

    # Kontext
    if kontext_latents is not None:
        image_ids = torch.concat([image_ids, kontext_image_ids], dim=-2) # (B, 2S, C), kontext image ids have the same sequence length S
        hidden_states = torch.concat([hidden_states, kontext_latents], dim=1) # (B, 2S, C), kontext latents have the same sequence length S

    hidden_states = dit.x_embedder(hidden_states)

    # TODO: batch dim expand to align with image_ids shape
    text_ids = repeat(text_ids, '1 ... -> b ...', b=image_ids.shape[0])

    prompt_emb = dit.context_embedder(prompt_emb)
    image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
    attention_mask = None

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, hidden_states, conditioning)
    else:
        tea_cache_update = False

    if tea_cache_update:
        hidden_states = tea_cache.update(hidden_states)
    else:
        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            num_frames = hidden_states.shape[0] if block_id < len(dit.blocks) // 3 else 1
            log(f"Double block {block_id} with number of frames: {num_frames}")
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                num_frames,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None),
            )
            # ControlNet
            if controlnet is not None and controlnet_conditionings is not None and controlnet_res_stack is not None:
                if kontext_latents is None:
                    hidden_states = hidden_states + controlnet_res_stack[block_id]
                else:
                    hidden_states[:, :-kontext_latents.shape[1]] = hidden_states[:, :-kontext_latents.shape[1]] + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            num_frames = hidden_states.shape[0] if block_id < len(dit.single_blocks) // 3 else 1
            log(f"Single block {block_id} with number of frames: {num_frames}")
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                num_frames,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None),
            )
            # ControlNet
            if controlnet is not None and controlnet_conditionings is not None and controlnet_single_res_stack is not None:
                if kontext_latents is None:
                    hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
                else:
                    hidden_states[:, prompt_emb.shape[1]:-kontext_latents.shape[1]] = hidden_states[:, prompt_emb.shape[1]:-kontext_latents.shape[1]] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)

    # Kontext
    if kontext_latents is not None:
        hidden_states = hidden_states[:, :-kontext_latents.shape[1]]

    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
