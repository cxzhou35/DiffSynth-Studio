import cv2
import torch
import numpy as np
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth.models.flux_vae_al import wrap_vae_with_al
from easyvolcap.utils.console_utils import *

def main(args):
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    pipe.enable_vram_management()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_image_dir = join(output_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    image_size = (args.width, args.height)
    if args.save_video:
        output_video_dir = join(output_dir, "videos")
        os.makedirs(output_video_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        orig_video_writer = cv2.VideoWriter(join(output_video_dir, "orig_video_60fps.mp4"), fourcc, 60, image_size)
        vae_type="al_vae" if args.use_al_vae else "orig_vae"
        recon_video_writer = cv2.VideoWriter(join(output_video_dir, f"{vae_type}_recon_video_60fps.mp4"), fourcc, 60, image_size)

    if args.use_al_vae:
        wrap_vae_with_al(pipe.vae_encoder, pipe.vae_decoder)
        pipe.vae_encoder.to(device=pipe.device, dtype=pipe.torch_dtype)
        pipe.vae_decoder.to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.load_models_to_device(["vae_encoder", "vae_decoder"])

    for image_file in tqdm(sorted(os.listdir(input_dir))):
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        assert image.size == image_size, f"Image size {image.size} does not match expected size {image_size}"
        input_image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)

        latents = pipe.vae_encoder(input_image, tiled=False)

        output_image = pipe.vae_decoder(latents, device=pipe.device, tiled=False)
        output_image = pipe.vae_output_to_image(output_image)

        output_path = os.path.join(output_image_dir, image_file)
        output_image.save(output_path)

        if args.save_video:
            # Write frames to video
            orig_video_writer.write(np.array(image)[:, :, ::-1])
            recon_video_writer.write(np.array(output_image)[:, :, ::-1])

    pipe.load_models_to_device([])

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test Flux VAE with Anti-Aliasing")
    # Add any arguments you want to pass to main here
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="",
        help="Path to the input image directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the images.",
    )
    parser.add_argument(
        "--save_video",
        default=False,
        action="store_true",
        help="Whether to save output as video.",
    )
    parser.add_argument(
        "--use_al_vae",
        default=False,
        action="store_true",
        help="Whether to use anti-aliased VAE.",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    main(args)
