import importlib
import torch, os, json
from diffsynth.pipelines.flux_image_new import ModelConfig, ControlNetInput
from diffsynth.pipelines.flux_4dsr import Flux4DSRPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.data.mvdata import MultiVideoDataset

from diffsynth.models.flux_vae_al import wrap_vae_with_al
from diffsynth.utils.base_utils import DotDict


class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        use_al_vae=False,
        use_al_dit=False,
        kontext_ref_offsets=None,
        use_fdl_loss=False,
        fdl_loss_weights=None,
        temporal_window_size=1,
        spatial_window_size=1,
        dit_3d_attn_interval=None,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = Flux4DSRPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Enable anti-aliased components in pipeline
        if use_al_vae:
            wrap_vae_with_al(self.pipe.vae_encoder, self.pipe.vae_decoder)
        if use_al_dit:
            assert False, "Anti-aliased DIT is not yet supported."

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.kontext_ref_offsets = kontext_ref_offsets if kontext_ref_offsets is not None else [1, 0, 0]
        self.use_fdl_loss = use_fdl_loss
        self.fdl_loss_weights = fdl_loss_weights if use_fdl_loss else None
        self.temporal_window_size= temporal_window_size
        self.spatial_window_size = spatial_window_size
        self.dit_3d_attn_interval = dit_3d_attn_interval

    def forward_preprocess(self, datas):
        # CFG-sensitive parameters
        # inputs_posi = {"prompt": data["prompt"]}
        inputs_posi = {"prompt": datas[0]["prompt"]}
        inputs_nega = {"negative_prompt": ""}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            # "input_image": data["image"],
            # "height": data["image"].size[1],
            # "width": data["image"].size[0],
            "input_image": [data["image"] for data in datas],
            "height": datas[0]["image"].size[1],
            "width": datas[0]["image"].size[0],
            "num_samples": len(datas),
            "kontext_ref_offsets": self.kontext_ref_offsets,
            # loss
            "use_fdl_loss": self.use_fdl_loss,
            "fdl_loss_weights": self.fdl_loss_weights,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "temporal_window_size": self.temporal_window_size,
            "spatial_window_size": self.spatial_window_size,
            "dit_3d_attn_interval": self.dit_3d_attn_interval,
        }

        # Extra inputs
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = [data[extra_input] for data in datas]
            else:
                inputs_shared[extra_input] = [data[extra_input] for data in datas]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss, loss_dict = self.pipe.training_loss(**models, **inputs)
        return loss, loss_dict


def main():
    # parse args
    parser = flux_parser()
    args = parser.parse_args()

    # create dataset from metadata
    datasets = DotDict()
    for split in ['train', 'val']:
        metadata_path = args.dataset_metadata_path.replace("train", split)
        dataset = MultiVideoDataset(
            base_path=args.dataset_base_path,
            metadata_path=metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            use_temporal_sample=args.use_temporal_sample,
            temporal_window_size=args.temporal_window_size,
            use_spatial_sample=args.use_spatial_sample,
            spatial_window_size=args.spatial_window_size,
            main_data_operator=MultiVideoDataset.default_image_operator(
                base_path=args.dataset_base_path,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            ),
            special_operator_map={
                "kontext_images": MultiVideoDataset.default_image_operator(
                    base_path=args.dataset_base_path,
                    max_pixels=args.max_pixels,
                    height_division_factor=16,
                    width_division_factor=16,
                ),
            }
        )
        datasets.update({split: dataset})

    # load model
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        use_al_vae=args.use_al_vae,
        use_al_dit=args.use_al_dit,
        kontext_ref_offsets=args.kontext_ref_offsets,
        use_fdl_loss=args.use_fdl_loss,
        fdl_loss_weights=args.fdl_loss_weights,
        temporal_window_size=args.temporal_window_size,
        spatial_window_size=args.spatial_window_size,
        dit_3d_attn_interval=args.dit_3d_attn_interval,
    )

    # set logger
    logging_dir = os.path.join("logs", os.path.basename(args.output_path))
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
    )

    launch_training_task(
        datasets,
        model,
        model_logger,
        logging_dir=logging_dir,
        mixed_precision="bf16",
        report_to="tensorboard",
        args=args
    )

def set_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONBREAKPOINT"] = "easyvolcap.utils.console_utils.set_trace"

if __name__ == "__main__":
    set_env()
    main()
