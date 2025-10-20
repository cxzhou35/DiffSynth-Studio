import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image
from diffusers.utils import load_image

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

input_image = load_image("/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p/43/000300.jpg")
input_image = input_image.resize((2560, 1440))

prompt = "Make this image highly detailed and realistic"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(
    prompt=prompt,
    kontext_images=input_image,
    embedded_guidance=2.5,
    seed=2,
)
image.save("old_tim_test_flux_kontext.jpg")
