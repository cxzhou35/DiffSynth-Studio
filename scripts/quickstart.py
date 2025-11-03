import torch
from diffsynth.pipelines.flux_4dsr import Flux4DSRPipeline, ModelConfig, ControlNetInput

pipe = Flux4DSRPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

image_1 = pipe(
    prompt="a beautiful Asian long-haired female college student.",
    embedded_guidance=2.5,
    seed=1,
)
image_1.save("image_1.jpg")

image_2 = pipe(
    prompt="transform the style to anime style.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=2,
)
image_2.save("image_2.jpg")
