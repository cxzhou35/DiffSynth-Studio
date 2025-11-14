import torch
from diffsynth.pipelines.flux_image_new import  ModelConfig
from diffsynth.pipelines.flux_4dsr import Flux4DSRPipeline
from PIL import Image
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_image", type=str, default="image_1.jpg", help="Condition image path or directory")
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--ckpt_path", type=str, default="models/train/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-4.safetensors")
    parser.add_argument("--output_dir", type=str, default="inference")
    parser.add_argument("--save_video", action="store_true", help="Whether save video")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--fps", type=int, default=60, help="FPS of video")
    return parser.parse_args()


def main(args):
    cond_image = args.cond_image
    prompt = args.prompt
    height = args.height
    width = args.width
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    fps = args.fps
    save_video = args.save_video
    video_path = args.video_path

    os.makedirs(output_dir, exist_ok=True)

    # load model
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
    pipe.load_lora(pipe.dit, ckpt_path, alpha=1)

    # pipe.enable_vram_management()

    if os.path.isdir(cond_image):
        cond_image_paths = [os.path.join(cond_image, f) for f in sorted(os.listdir(cond_image)) if f.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        cond_image_paths = [cond_image]

    video_writer = None
    if save_video and video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_path in tqdm(cond_image_paths, desc="Inferring with Flux.1-dev-Kontext-LoRA-Finetune", total=len(cond_image_paths)):
        cond_image = Image.open(image_path)
        cond_image = cond_image.resize((width, height))
        # inference
        result_image = pipe(
            prompt=prompt,
            kontext_images=cond_image,
            height=height, width=width,
            embedded_guidance=3.5,
            seed=0,
        )

        # save results
        # image_save_dir = os.path.join(output_dir, "images")
        # os.makedirs(image_save_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        # concat the control image and result image side by side
        concat_image = Image.new('RGB', (width * 2, height))
        concat_image.paste(cond_image, (0, 0))
        concat_image.paste(result_image, (width, 0))
        concat_image.save(save_path)
        if video_writer is not None:
            video_writer.write(cv2.cvtColor(np.array(concat_image), cv2.COLOR_RGB2BGR))

    if video_writer is not None:
        video_writer.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)
