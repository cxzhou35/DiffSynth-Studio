import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth.models.flux_vae_al import wrap_vae_with_al


# pipe = FluxImagePipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
#         ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
#         ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
#         ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
#     ],
# )

# wrap_vae_with_al(pipe.vae)

# prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
# negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

# image = pipe(prompt=prompt, seed=0)
# image.save("flux.jpg")

# image = pipe(
#     prompt=prompt, negative_prompt=negative_prompt,
#     seed=0, cfg_scale=2, num_inference_steps=50,
# )
# image.save("flux_cfg.jpg")

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

wrap_vae_with_al(pipe.vae_encoder, pipe.vae_decoder)

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

image_3 = pipe(
    prompt="let her smile.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=3,
)
image_3.save("image_3.jpg")
