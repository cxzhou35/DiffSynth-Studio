import argparse
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.flux_4dsr import ControlNetInput, Flux4DSRPipeline
from diffsynth.pipelines.flux_image_new import ModelConfig

# Optional: use diffusers image loader when available.
try:
    from diffusers.utils import load_image as diffusers_load_image

    HAS_DIFFUSERS = True
except Exception:
    diffusers_load_image = None
    HAS_DIFFUSERS = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_image', type=str, default='image_1.jpg', help='Condition image path or directory')
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='models/train/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-4.safetensors',
    )
    parser.add_argument('--output_dir', type=str, default='inference')
    parser.add_argument('--save_video', action='store_true', help='Whether save video')
    parser.add_argument('--fps', type=int, default=60, help='FPS of video')
    parser.add_argument('--relative_frame_range', nargs=3, type=int, help='Relative frame ranges')
    parser.add_argument('--view_range', nargs=3, type=int, help='View ranges')
    parser.add_argument('--use_3d_attn', action='store_true', help='Whether use 3D attention')
    parser.add_argument('--attn_window_size', type=int, default=3, help='3D attention window size')
    parser.add_argument('--use_3d_rope', action='store_true', help='Whether use 3D RoPE')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument(
        '--denoising_strategy',
        type=str,
        default='standard',
        choices=['standard', 'sliding_iterative'],
        help='Denoising strategy: standard or sliding_iterative',
    )

    # Sliding iterative denoising (Diffuman4D-style window scheduling)
    parser.add_argument('--use_sliding_iterative_denoise', action='store_true')
    parser.add_argument('--sliding_window_size', type=int, default=4)
    parser.add_argument('--sliding_window_stride', type=int, default=1)
    parser.add_argument('--sliding_window_shift', type=int, default=0)
    parser.add_argument('--sliding_bidirectional', action='store_true')
    parser.add_argument('--sliding_num_denoising_steps', type=int, default=1)
    parser.add_argument('--sliding_alternation_rounds', type=int, default=0)

    args = parser.parse_args()
    # Backward compatibility for existing launch scripts.
    if args.use_sliding_iterative_denoise:
        args.denoising_strategy = 'sliding_iterative'
    return args

def split_even(items: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [items]
    base = len(items) // n
    rem = len(items) % n
    chunks = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def select_view_dirs(cond_image: str, view_range: List[int]) -> List[str]:
    vs, ve, vt = view_range
    all_entries = sorted(os.listdir(cond_image))

    if ve >= len(all_entries):
        raise ValueError(
            f"view_range end={ve} out of bounds for {len(all_entries)} entries under {cond_image}"
        )

    view_dirs = []
    for idx in range(vs, ve + 1, vt):
        path = os.path.join(cond_image, all_entries[idx])
        if os.path.isdir(path):
            view_dirs.append(all_entries[idx])
    return view_dirs


def list_view_image_paths(view_dir_path: str, relative_frame_range: Optional[List[int]]) -> List[str]:
    frame_files = sorted(os.listdir(view_dir_path))
    if relative_frame_range:
        s, e, t = relative_frame_range
        frame_files = frame_files[s : e + 1 : t]
    return [os.path.join(view_dir_path, frame_file) for frame_file in frame_files]


def get_video_dict(save_dir, view_dirs, width, height, fps):
    video_list = ["cond", "pred", "concat"]
    video_dict = {}
    for view_dir in view_dirs:
        for video_name in video_list:
            video_save_path = os.path.join(save_dir, view_dir, f"{video_name}.mp4")
            if video_name == "concat":
                video_width, video_height = width * 2, height
            else:
                video_width, video_height = width, height
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_dict.update(
                {
                    f"{view_dir}_{video_name}": cv2.VideoWriter(
                        video_save_path, fourcc, fps, (video_width, video_height)
                    )
                }
            )
    return video_dict


def load_cond_image(img_path: str) -> Image.Image:
    # Prefer diffusers loader if available.
    if HAS_DIFFUSERS and diffusers_load_image is not None:
        try:
            return diffusers_load_image(img_path).convert("RGB")
        except Exception:
            pass
    return Image.open(img_path).convert("RGB")


def main(args):
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

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
    attn_window_size = args.attn_window_size
    use_3d_rope = args.use_3d_rope
    num_inference_steps = args.num_inference_steps

    image_save_dir = os.path.join(output_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)

    assert os.path.exists(cond_image), f"Condition image path {cond_image} does not exist."

    view_dirs = None
    view_sequences: List[List[str]] = []

    if relative_frame_range and view_range and os.path.isdir(cond_image):
        all_view_dirs = select_view_dirs(cond_image, view_range)
        view_shards = split_even(all_view_dirs, world_size)
        view_dirs = view_shards[rank] if rank < len(view_shards) else []

        if is_main:
            print(f"[INFO] Accelerator world_size={world_size}")
            print(f"[INFO] Selected total views={len(all_view_dirs)}")
        print(f"[RANK {rank}] assigned views={len(view_dirs)} -> {view_dirs}")

        if len(view_dirs) == 0:
            print(f"[RANK {rank}] no assigned views, exiting early.")
            accelerator.wait_for_everyone()
            return

        for view_dir in view_dirs:
            view_dir_path = os.path.join(cond_image, view_dir)
            if not os.path.isdir(view_dir_path):
                continue
            image_paths = list_view_image_paths(view_dir_path, relative_frame_range)
            if len(image_paths) > 0:
                view_sequences.append(image_paths)

        cond_image_paths = [p for seq in view_sequences for p in seq]
        num_images = len(cond_image_paths)
        num_views = len(view_sequences)
        num_frames = num_images // num_views if num_views > 0 else 0
        print(f"total {num_images} images found, {num_views} views, {num_frames} frames per view.")

    elif os.path.isdir(cond_image):
        all_files = [
            os.path.join(cond_image, f)
            for f in sorted(os.listdir(cond_image))
            if os.path.isfile(os.path.join(cond_image, f))
        ]
        file_shards = split_even(all_files, world_size)
        cond_image_paths = file_shards[rank] if rank < len(file_shards) else []

        if is_main:
            print(f"[INFO] Accelerator world_size={world_size}")
            print(f"[INFO] Selected total files={len(all_files)}")
        print(f"[RANK {rank}] assigned files={len(cond_image_paths)}")

        if len(cond_image_paths) == 0:
            print(f"[RANK {rank}] no assigned files, exiting early.")
            accelerator.wait_for_everyone()
            return

        view_sequences = [cond_image_paths]

    else:
        cond_image_paths = [cond_image]
        view_sequences = [cond_image_paths]
        if rank > 0:
            # Single-image mode: only rank0 should run to avoid duplicate outputs.
            accelerator.wait_for_everyone()
            return

    if save_video:
        if view_dirs is None:
            print(f"[RANK {rank}] save_video requested without view_dirs; disable video writing.")
            save_video = False
        else:
            video_save_dir = os.path.join(output_dir, "videos")
            video_dict = get_video_dict(video_save_dir, view_dirs, width, height, fps)

    model_device = str(accelerator.device)
    print(f"[RANK {rank}] loading model on device={model_device}")

    pipe = Flux4DSRPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=model_device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
            ModelConfig(
                model_id="jasperai/Flux.1-dev-Controlnet-Upscaler",
                origin_file_pattern="diffusion_pytorch_model.safetensors",
            ),
        ],
    )
    pipe.load_lora(pipe.dit, ckpt_path, alpha=1)

    use_sliding_iterative = args.denoising_strategy == 'sliding_iterative'

    if use_sliding_iterative:
        # Sliding iterative works on full sequence per view.
        inference_batches = view_sequences
        mode_desc = "sliding-iterative"
    else:
        inference_batches = [
            cond_image_paths[i : i + attn_window_size]
            for i in range(0, len(cond_image_paths), attn_window_size)
        ]
        mode_desc = "fixed-window"

    for image_paths in tqdm(
        inference_batches,
        desc=f"[RANK {rank}] Inferring ({mode_desc}) with Flux.1-dev-Controlnet-Upscaler",
        total=len(inference_batches),
    ):
        cond_images = []
        for img_path in image_paths:
            img = load_cond_image(img_path)
            img = img.resize((width, height))
            cond_images.append(img)

        if use_sliding_iterative:
            if len(cond_images) == 0:
                continue
            sliding_window_size = min(args.sliding_window_size, len(cond_images))
            sliding_window_stride = args.sliding_window_stride
            if len(cond_images) % sliding_window_stride != 0:
                print(
                    f"[RANK {rank}] WARN: len(cond_images)={len(cond_images)} is not divisible by "
                    f"sliding_window_stride={sliding_window_stride}; fallback stride=1"
                )
                sliding_window_stride = 1

            sliding_alternation_rounds = (
                args.sliding_alternation_rounds if args.sliding_alternation_rounds > 0 else None
            )

            pred_images = pipe.sliding_iterative_denoise(
                prompt=prompt,
                controlnet_inputs=[ControlNetInput(images=cond_images, scale=0.9)],
                height=height,
                width=width,
                seed=0,
                rand_device="cuda",
                num_inference_steps=num_inference_steps,
                num_samples=len(cond_images),
                dit_3d_attn_interval=3,
                use_3d_rope=use_3d_rope,
                tiled=True,
                tile_size=128,
                tile_stride=64,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                sliding_window_shift=args.sliding_window_shift,
                sliding_bidirectional=args.sliding_bidirectional,
                sliding_num_denoising_steps=args.sliding_num_denoising_steps,
                sliding_alternation_rounds=sliding_alternation_rounds,
            )
        else:
            pred_images = pipe(
                prompt=prompt,
                controlnet_inputs=[ControlNetInput(images=cond_images, scale=0.9)],
                height=height,
                width=width,
                seed=0,
                rand_device="cuda",
                num_inference_steps=num_inference_steps,
                num_samples=len(cond_images),
                dit_3d_attn_interval=3,
                use_3d_rope=use_3d_rope,
                tiled=True,
                tile_size=128,
                tile_stride=64,
            )

        if not isinstance(pred_images, list):
            pred_images = [pred_images]

        for idx, img_path in enumerate(image_paths):
            image_file = os.path.basename(img_path)
            view_dir = os.path.basename(os.path.dirname(img_path))
            pred_image_save_path = os.path.join(image_save_dir, "pred", view_dir, image_file)
            concat_image_save_path = os.path.join(image_save_dir, "concat", view_dir, image_file)
            os.makedirs(os.path.dirname(pred_image_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(concat_image_save_path), exist_ok=True)

            concat_image = Image.new("RGB", (width * 2, height))
            concat_image.paste(cond_images[idx], (0, 0))
            concat_image.paste(pred_images[idx], (width, 0))

            pred_images[idx].save(pred_image_save_path)
            concat_image.save(concat_image_save_path)

            if save_video:
                pred_frame = cv2.cvtColor(np.array(pred_images[idx], dtype=np.uint8), cv2.COLOR_RGB2BGR)
                concat_frame = cv2.cvtColor(np.array(concat_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                video_dict[f"{view_dir}_pred"].write(pred_frame)
                video_dict[f"{view_dir}_concat"].write(concat_frame)

    if save_video:
        for video_writer in video_dict.values():
            video_writer.release()

    accelerator.wait_for_everyone()
    if is_main:
        print("[OK] Inference finished on all accelerator processes.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
