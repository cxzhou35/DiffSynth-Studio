import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..schedulers import FlowMatchScheduler
from ..prompters import FluxPrompter
from ..models import ModelManager, load_state_dict, SD3TextEncoder1, FluxTextEncoder2, FluxDiT, FluxVAEEncoder, FluxVAEDecoder
from ..models.flux_ipadapter import FluxIpAdapter
from ..models.flux_infiniteyou import InfiniteYouImageProjector
from ..models.flux_lora_encoder import FluxLoRAEncoder, LoRALayerBlock
from ..models.tiler import FastTileWorker
from ..utils import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit
from ..lora.flux_lora import FluxLoRALoader, FluxLoraPatcher, FluxLoRAFuser

from ..models.flux_dit import RMSNorm
from ..vram_management import gradient_checkpoint_forward, enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from .flux_image_new import (
    ControlNetInput,
    MultiControlNet,
    FluxImageUnit_ShapeChecker,
    FluxImageUnit_NoiseInitializer,
    FluxImageUnit_PromptEmbedder,
    FluxImageUnit_InputImageEmbedder,
    FluxImageUnit_ImageIDs,
    FluxImageUnit_EmbeddedGuidanceEmbedder,
    FluxImageUnit_Kontext,
    FluxImageUnit_InfiniteYou,
    FluxImageUnit_ControlNet,
    FluxImageUnit_IPAdapter,
    FluxImageUnit_EntityControl,
    FluxImageUnit_TeaCache,
    FluxImageUnit_LoRAEncode,
    InfinitYou,
    TeaCache,
)


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
        self.infinityou_processor: InfinitYou = None
        self.image_proj_model: InfiniteYouImageProjector = None
        self.lora_patcher: FluxLoraPatcher = None
        self.lora_encoder: FluxLoRAEncoder = None
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
            FluxImageUnit_InfiniteYou(),
            FluxImageUnit_ControlNet(),
            FluxImageUnit_IPAdapter(),
            FluxImageUnit_EntityControl(),
            FluxImageUnit_TeaCache(),
            FluxImageUnit_LoRAEncode(),
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
                    LoRALayerBlock: AutoWrappedModule,
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
        default_vram_management_models = ["text_encoder_1", "vae_decoder", "vae_encoder", "controlnet", "image_proj_model", "ipadapter", "lora_patcher", "lora_encoder"]
        for model_name in default_vram_management_models:
            self._enable_vram_management_with_default_config(getattr(self, model_name), vram_limit)

        # Special config
        if self.text_encoder_2 is not None:
            from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseActDense, T5DenseGatedActDense
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
            from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings, SiglipEncoder, SiglipMultiheadAttentionPoolingHead
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
        pipe.image_proj_model = model_manager.fetch_model("infiniteyou_image_projector")
        if pipe.image_proj_model is not None:
            pipe.infinityou_processor = InfinitYou(device=device)
        pipe.lora_patcher = model_manager.fetch_model("flux_lora_patcher")
        pipe.lora_encoder = model_manager.fetch_model("flux_lora_encoder")

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
        # ControlNet
        controlnet_inputs: list[ControlNetInput] = None,
        # IP-Adapter
        ipadapter_images: Union[list[Image.Image], Image.Image] = None,
        ipadapter_scale: float = 1.0,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        eligen_enable_inpaint: bool = False,
        # InfiniteYou
        infinityou_id_image: Image.Image = None,
        infinityou_guidance: float = 1.0,
        # LoRA Encoder
        lora_encoder_inputs: Union[list[ModelConfig], ModelConfig, str] = None,
        lora_encoder_scale: float = 1.0,
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
            "controlnet_inputs": controlnet_inputs,
            "ipadapter_images": ipadapter_images, "ipadapter_scale": ipadapter_scale,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative, "eligen_enable_inpaint": eligen_enable_inpaint,
            "infinityou_id_image": infinityou_id_image, "infinityou_guidance": infinityou_guidance,
            "lora_encoder_inputs": lora_encoder_inputs, "lora_encoder_scale": lora_encoder_scale,
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
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    id_emb=None,
    infinityou_guidance=None,
    flex_condition=None,
    flex_uncondition=None,
    flex_control_stop_timestep=None,
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
        if id_emb is not None:
            controlnet_text_ids = torch.zeros(id_emb.shape[0], id_emb.shape[1], 3).to(device=hidden_states.device, dtype=hidden_states.dtype)
            controlnet_extra_kwargs.update({"prompt_emb": id_emb, 'text_ids': controlnet_text_ids, 'guidance': infinityou_guidance})
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_conditionings, **controlnet_extra_kwargs
        )

    # Flex
    if flex_condition is not None:
        if timestep.tolist()[0] >= flex_control_stop_timestep:
            hidden_states = torch.concat([hidden_states, flex_condition], dim=1)
        else:
            hidden_states = torch.concat([hidden_states, flex_uncondition], dim=1)

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)

    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)

    # Kontext
    if kontext_latents is not None:
        image_ids = torch.concat([image_ids, kontext_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, kontext_latents], dim=1)

    hidden_states = dit.x_embedder(hidden_states)

    # EliGen
    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids, latents.shape[1])
    else:
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
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
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
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
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
