import torch
from modelscope import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p/43/000300.jpg")

input_image = input_image.resize((2560, 1440))

prompt = "Make this image highly detailed and realistic"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(
  image=input_image,
  prompt=prompt,
  guidance_scale=2.5
).images[0]

image.save("old_tim_test_flux_kontext.jpg")
