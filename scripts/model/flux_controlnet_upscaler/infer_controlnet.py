import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from PIL import Image
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join, exists

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_image", type=str, default="image_1.jpg", help="Condition image path or directory")
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--ckpt_path", type=str, default="models/train/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-4.safetensors")
    parser.add_argument("--output_dir", type=str, default="inference")
    parser.add_argument("--save_video", action="store_true", help="Whether save video")
    parser.add_argument("--fps", type=int, default=60, help="FPS of video")
    parser.add_argument("--relative_frame_range", nargs=3, type=int, help="Relative frame ranges")
    parser.add_argument("--view_range", nargs=3, type=int, help="View ranges")
    return parser.parse_args()

def get_evc_paths(cond_image, relative_frame_range, view_range):
    vs, ve, vt = view_range
    view_dirs = [f"{dir:02d}" for dir in range(vs, ve+1, vt)]
    # view_dirs = sorted(os.listdir(cond_image))[vs:ve+1:vt]
    image_paths = []
    view_dirs_copy = view_dirs.copy()
    for view_dir in view_dirs_copy:
        view_dir_path = os.path.join(cond_image, view_dir)
        if not os.path.exists(view_dir_path):
            view_dirs.remove(view_dir)
            continue
        frame_files = sorted(os.listdir(view_dir_path))
        if relative_frame_range:
            s, e, t = relative_frame_range
            frame_files = frame_files[s:e+1:t]
        for frame_file in frame_files:
            image_paths.append(os.path.join(view_dir_path, frame_file))
    num_images = len(image_paths)
    num_views = len(view_dirs)
    num_frames = num_images // num_views
    print(f"total {num_images} images found, {num_views} views, {num_frames} frames per view.")
    return num_images, image_paths, view_dirs

def get_video_dict(save_dir, view_dirs, width, height, fps):
    video_list = ['cond', 'pred', 'concat']
    video_dict = {}
    for view_dir in view_dirs:
        for video_name in video_list:
            video_save_path = os.path.join(save_dir, view_dir, f"{video_name}.mp4")
            if video_name == 'concat':
                video_width, video_height = width*2, height
            else:
                video_width, video_height = width, height
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_dict.update({
                f"{view_dir}_{video_name}": cv2.VideoWriter(video_save_path, fourcc, fps, (video_width, video_height))
            })
    return video_dict

def main(args):
    cond_image = args.cond_image
    prompt = args.prompt
    height = args.height
    width = args.width
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    fps = args.fps
    save_video = args.save_video
    relative_frame_range = args.relative_frame_range
    view_range = args.view_range

    image_save_dir = os.path.join(output_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)

    assert os.path.exists(cond_image), f"Condition image path {cond_image} does not exist."
    if relative_frame_range and view_range:
        num_images, cond_image_paths, view_dirs = get_evc_paths(cond_image, relative_frame_range, view_range)
    elif os.path.isdir(cond_image):
        cond_image_paths = [os.path.join(cond_image, f) for f in sorted(os.listdir(cond_image))]
    else:
        cond_image_paths = [cond_image]
    img_ext = os.path.splitext(cond_image_paths[0])[1]

    if save_video:
        video_save_dir = os.path.join(output_dir, "videos")
        video_dict = get_video_dict(video_save_dir, view_dirs, width, height, fps)

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

    for image_path in tqdm(cond_image_paths, desc="Inferring with Flux.1-dev-Controlnet-Upscaler", total=len(cond_image_paths)):
        cond_image = Image.open(image_path)
        cond_image = cond_image.resize((width, height))

        image_file = os.path.basename(image_path)
        view_dir = os.path.basename(os.path.dirname(image_path))
        pred_image_save_path = os.path.join(image_save_dir, "pred", view_dir, image_file)
        concat_image_save_path = os.path.join(image_save_dir, "concat", view_dir, image_file)
        os.makedirs(os.path.dirname(pred_image_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(concat_image_save_path), exist_ok=True)

        # model inference
        pred_image = pipe(
            prompt=prompt,
            controlnet_inputs=[ControlNetInput(
                images=cond_image,
                scale=0.9
            )],
            height=height, width=width,
            seed=0, rand_device="cuda",
        )

        concat_image = Image.new('RGB', (width * 2, height))
        concat_image.paste(cond_image, (0, 0))
        concat_image.paste(pred_image, (width, 0))

        # save images
        pred_image.save(pred_image_save_path)
        concat_image.save(concat_image_save_path)

        if save_video:
            # ensure uint8 RGB before writing to avoid malformed mp4
            pred_frame = cv2.cvtColor(np.array(pred_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            concat_frame = cv2.cvtColor(np.array(concat_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            video_dict[f'{view_dir}_pred'].write(pred_frame)
            video_dict[f'{view_dir}_concat'].write(concat_frame)

    if save_video:
        for video_writer in video_dict.values():
            video_writer.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)
