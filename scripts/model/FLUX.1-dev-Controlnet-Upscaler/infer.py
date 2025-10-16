import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from PIL import Image
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_image", type=str, default="image_1.jpg", help="Control image path or directory")
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--ckpt_path", type=str, default="models/train/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-4.safetensors")
    parser.add_argument("--output_dir", type=str, default="inference")
    parser.add_argument("--save_video", action="store_true", help="Whether save video")
    parser.add_argument("--fps", type=int, default=60, help="FPS of video")
    return parser.parse_args()


def main(args):
    control_image = args.control_image
    prompt = args.prompt
    height = args.height
    width = args.width
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    fps = args.fps

    os.makedirs(output_dir, exist_ok=True)

    # load model
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
            ModelConfig(model_id="jasperai/Flux.1-dev-Controlnet-Upscaler", origin_file_pattern="diffusion_pytorch_model.safetensors"),
        ],
    )
    pipe.load_lora(pipe.dit, ckpt_path, alpha=1)

    if os.path.isdir(control_image):
        control_image_paths = [os.path.join(control_image, f) for f in sorted(os.listdir(control_image)) if f.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        control_image_paths = [control_image]

    if args.save_video:
        video_save_dir = os.path.join(output_dir, 'videos')
        os.makedirs(video_save_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(os.path.join(video_save_dir, f'video_{fps:02d}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_path in tqdm(control_image_paths, desc="Inferring with Flux.1-dev-Controlnet-Upscaler", total=len(control_image_paths)):
        control_image = Image.open(image_path)
        control_image = control_image.resize((width, height))
        # inference
        image = pipe(
            prompt=prompt,
            controlnet_inputs=[ControlNetInput(
                image=control_image,
                scale=0.9
            )],
            height=height, width=width,
            seed=0, rand_device="cuda",
        )

        # save results
        # image_save_dir = os.path.join(output_dir, "images")
        # os.makedirs(image_save_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        print(f"Save result to {save_path}")
        image.save(save_path)
        if args.save_video:
            video_writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    if args.save_video:
        video_writer.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)
