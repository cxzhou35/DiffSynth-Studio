# triangulate each pair images sequentially and fuse them for every frame in evc file format
import argparse
import ast
import os
import queue
import sys
import time
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from os.path import abspath, dirname, isabs, isdir, isfile, join

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from romav2 import RoMaV2
from tqdm import tqdm

from utils.data_utils import export_pts
from utils.easy_utils import read_camera

device = torch.device("cuda")

try:
    from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor
    model_dict = {
        "indoor": roma_indoor,
        "outdoor": roma_outdoor,
        "tiny": tiny_roma_v1_outdoor,
        "v2": RoMaV2,
    }
except ImportError:
    print("romatch not found, using romav2 instead")
    from romav2 import RoMaV2
    model_dict = {
        "v2": RoMaV2,
    }


class AsyncTaskExecutor:
    """Generic asynchronous task executor for running functions in background threads

    Examples:
        # File operations
        executor.submit_task(export_pts, filename, points, color=colors)

        # Image processing
        executor.submit_task(cv2.imwrite, 'output.jpg', image)

        # Data processing
        executor.submit_task(np.save, 'data.npy', array)

        # Custom functions
        executor.submit_task(my_custom_function, arg1, arg2, keyword=value)
    """

    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.pending_tasks = []
        self.shutdown_event = threading.Event()

    def submit_task(self, func, *args, **kwargs):
        """Submit a task (function call) to be executed asynchronously"""
        future = self.executor.submit(self._execute_task, func, *args, **kwargs)
        self.pending_tasks.append(future)
        return future

    def _execute_task(self, func, *args, **kwargs):
        """Internal method to execute the task function"""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error executing task {func.__name__}: {e}")
            return None

    def wait_for_completion(self):
        """Wait for all pending tasks to complete"""
        for future in self.pending_tasks:
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                print(f"Task execution failed: {e}")
        self.pending_tasks.clear()

    def get_pending_count(self):
        """Get the number of pending task operations"""
        return len([f for f in self.pending_tasks if not f.done()])

    def shutdown(self):
        """Shutdown the task executor and wait for completion"""
        self.wait_for_completion()
        self.executor.shutdown(wait=True)


class SequentialImageCache:
    """Used to prefetch images whose access order is known in advance"""

    def __init__(self, max_size=50, max_workers=4):
        self.cache = {}
        self.max_size = max_size
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.pending_paths = queue.Queue()  # Queue for paths that need to be loaded
        self.target_size = None
        # For eviction if some images are skipped
        self.access_sequence = []
        self.path_to_last_position = {}  # path -> latest position it appears
        self.current_pos = 0
        self.skip_threshold = 10  # positions past due before considering skipped

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        # Worker thread
        self.prefetch_thread = None
        self.shutdown_event = threading.Event()
        self._start_prefetch_worker()

    def add_paths(self, image_paths, target_size):
        """Add image paths to the pending queue for worker thread processing"""
        self.target_size = target_size

        # Add unique paths to pending queue for prefetching (preserve order)
        unique_paths = []
        for path in image_paths:
            if path not in unique_paths:
                unique_paths.append(path)

        for path in unique_paths:
            self.pending_paths.put(path)

        self.access_sequence = image_paths
        self.path_to_last_position = {}
        for i, path in enumerate(image_paths):
            self.path_to_last_position[path] = i

        print(f"Cache initialized: {len(image_paths)} accesses, {len(unique_paths)} unique images")

        time.sleep(0.5)  # prefetch the first few images before using

    def _start_prefetch_worker(self):
        """Start the persistent prefetch worker thread"""

        def prefetch_worker():
            while not self.shutdown_event.is_set():
                try:
                    with self.lock:
                        available_slots = self.max_size - len(self.cache)

                    if available_slots > 0:
                        paths_to_prefetch = []
                        for _ in range(available_slots):
                            try:
                                path = self.pending_paths.get_nowait()
                                with self.lock:
                                    if path not in self.cache:
                                        paths_to_prefetch.append(path)
                            except queue.Empty:
                                break

                        if paths_to_prefetch:
                            self._prefetch_batch(paths_to_prefetch, self.target_size)

                    # If no space or no paths, sleep briefly before checking again
                    if available_slots == 0 or self.pending_paths.empty():
                        threading.Event().wait(0.1)  # Sleep for 100ms

                except Exception as e:
                    print(f"Error in prefetch worker: {e}")
                    self.shutdown()

        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _prefetch_batch(self, image_paths, target_size):
        tasks = []

        for img_path in image_paths:
            last_chance_pos = self.path_to_last_position[img_path]
            if self._get_cache(img_path) is None and last_chance_pos > self.current_pos:
                future = self.executor.submit(load_and_resize_image, img_path, target_size)
                tasks.append((future, img_path))

        for future, img_path in tasks:
            try:
                result = future.result()
                if result[0] is not None:  # Check if image loaded successfully
                    self._put_cache(img_path, result)
            except Exception as e:
                print(f"Error prefetching image for {img_path}: {e}")

    def _get_cache(self, key):
        with self.lock:
            return self.cache.get(key)

    def _put_cache(self, key, value):
        with self.lock:
            self.cache[key] = value

    def get_image_or_load(self, image_path, target_size=None):
        """Get image from cache or load directly if not cached"""
        cached_result = self._get_cache(image_path)

        if cached_result is not None:
            result = cached_result
        else:
            # Cache miss - load image directly
            print(f"Cache miss, loading image {image_path} from disk")
            if target_size is None:
                target_size = self.target_size
            result = load_and_resize_image(image_path, target_size)

        self._cleanup_overdue_images(image_path)

        return result

    def shutdown(self):
        self.shutdown_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
        self.executor.shutdown(wait=False)

    def _cleanup_overdue_images(self, accessed_path):
        """Remove images whose last chance to be accessed has passed"""
        if accessed_path not in self.path_to_last_position:
            return

        with self.lock:
            self.current_pos = self.access_sequence.index(accessed_path, self.current_pos)

            to_remove = []
            for cached_path in list(self.cache.keys()):  # Check all items since the cache size is relatively small
                last_chance_pos = self.path_to_last_position[cached_path]
                if self.current_pos > last_chance_pos + self.skip_threshold:
                    to_remove.append(cached_path)

            for path in to_remove:
                self.cache.pop(path, None)


def load_and_resize_image(image_path, target_size):
    """Load and resize image, return tuple of (resized_image, original_width, original_height)"""
    try:
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        img_resized = img.resize(target_size) if target_size is not None else img
        return img_resized, orig_w, orig_h
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None


def construct_access_sequence(frame_list, view_list, image_dir):
    """Construct the actual access sequence matching the processing order"""
    access_sequence = []
    for frame in frame_list:
        for view1, view2 in view_list:
            for view in [view1, view2]:
                img_path = join(image_dir, view, frame)
                access_sequence.append(img_path)
    return access_sequence


def parse_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid Python list input: {string}")


@torch.jit.script
def triangulate_nviews_batch(P, ip):
    """
    Triangulate points visible in n camera views using PyTorch on CUDA without for loop.
    P is a tensor of shape (b, n, 3, 4) where b is batch size and n is number of views.
    ip is a tensor of shape (b, n, 3).
    """
    batch_size, num_views, _, _ = P.shape

    # Create the large M matrix in a batched manner
    M = torch.zeros((batch_size, 3 * num_views, 4 + num_views), device=P.device)  # b, 3 * 2, 4 + 2

    for i in range(num_views):
        M[:, 3 * i : 3 * i + 3, :4] = P[:, i, :, :]  # b, 3, 4
        M[:, 3 * i : 3 * i + 3, 4 + i] = -ip[:, i, :]  # b, 3, 1

    U, S, V = torch.linalg.svd(M)
    X = V[:, -1, :4]
    X = X / X[:, 3].unsqueeze(-1)

    return X


@torch.jit.script
def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in x1, x2 (nx3 homog. coordinates) using PyTorch on CUDA.
    P1 and P2 are tensors of shape (3, 4).
    x1 and x2 are tensors of shape (n, 3).
    """
    if x1.shape != x2.shape:
        raise ValueError("Number of points don't match.")

    # Stack P1 and P2 into a single tensor
    P = torch.stack([P1, P2], dim=0).unsqueeze(0).repeat(x1.shape[0], 1, 1, 1)  # b, 2, 3, 4

    # Stack x1 and x2 into a single tensor
    ip = torch.stack([x1, x2], dim=1)  # (b, 2, 3)

    X = triangulate_nviews_batch(P, ip)

    return X


def roma_triangulate(
    roma_model: nn.Module,
    im1: Image.Image,
    im2: Image.Image,
    ori_size1: tuple[int, int],
    ori_size2: tuple[int, int],
    H: int,
    W: int,
    cam1: dict,
    cam2: dict,
    model_type: str,
    mask_dir: Optional[str] = None,
    certainty_thresh: float = 0.2,
    device: Optional[torch.device] = torch.device("cuda"),
    frame_name: Optional[str] = None,
):
    """
    Triangulate points from two images using RoMA model.

    Args:
        roma_model: RoMA model instance
        im1, im2: PIL Image objects
        H, W: Target resolution
        cams: Camera parameters dictionary
        view1, view2: View names for camera parameters
        model_type: Model type string
        device: torch device
        mask_dir: Optional mask directory path
        certainty_thresh: Certainty threshold for filtering matches
        frame_name: Optional frame name for mask loading and logging

    Returns:
        point: Triangulated 3D points (N, 3)
        colors: Corresponding RGB colors (N, 3)
    """
    im1_H, im1_W = ori_size1
    im2_H, im2_W = ori_size2

    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)  # (3, H, W)

    # Match
    if model_type == "v2":
        preds = roma_model.match(im1, im2)
        warp = preds["warp_AB"][0]
        certainty = preds["overlap_AB"][0]
        # Sample RGB values from warped coordinates
        sampled_rgb = F.grid_sample(x2[None], warp[None], mode="bilinear", align_corners=False)
        im2_transfer_rgb = sampled_rgb[0].permute(1, 2, 0)  # (H, W, 3)
        certainty = certainty[:H, :W]
        assert warp.shape[:2] == (H, W)
    else:
        if model_type == "tiny_outdoor":
            warp, certainty = roma_model.match(im1, im2)  # (H, W, 4), (H, W)
        else:
            warp, certainty = roma_model.match(im1, im2, device=device)  # (H, 2*W, 4), (H, 2*W)
            if warp.ndim == 4 and warp.shape[0] == 1:
                warp, certainty = warp[0], certainty[0]  # romatch 0.1.2
        # Sample RGB values from warped coordinates
        warped_coords = warp[:, :W, 2:][None]
        sampled_rgb = F.grid_sample(x2[None], warped_coords, mode="bilinear", align_corners=False)
        im2_transfer_rgb = sampled_rgb[0].permute(1, 2, 0)  # (H, W, 3)
        certainty = certainty[:, :W]

    # Create coordinate grid for image 1
    y_coords = torch.arange(H, device=device)
    x_coords = torch.arange(W, device=device)
    img1_corrs = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1).flip(-1)  # (H, W, 2)
    if model_type == "v2":
        img2_corrs = warp[:H, :W, :2].clone()
    else:
        img2_corrs = warp[:, :W, 2:].clone()
    img2_corrs[:, :, 0] = (img2_corrs[:, :, 0] + 1) * (W - 1) / 2
    img2_corrs[:, :, 1] = (img2_corrs[:, :, 1] + 1) * (H - 1) / 2

    if mask_dir and frame_name:
        mask_path = join(mask_dir, view1, frame_name.replace(".jpg", ".png"))
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask.resize((W, H)))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_index = torch.from_numpy(mask > 0).to(device)
        img1_corrs = img1_corrs[mask_index]
        img2_corrs = img2_corrs[mask_index]
        im2_transfer_rgb = im2_transfer_rgb[mask_index]
        certainty = certainty[mask_index]

    img1_corrs, img2_corrs = img1_corrs.reshape(-1, 2), img2_corrs.reshape(-1, 2)  # (H*W, 2)
    valid_index = certainty.reshape(-1) > certainty_thresh
    img1_corrs, img2_corrs = img1_corrs[valid_index], img2_corrs[valid_index]  # (N, 2)
    colors = im2_transfer_rgb.reshape(-1, 3)[valid_index]
    if frame_name:
        print(f"num of matches between {view1} and {view2} ({frame_name}): {len(img1_corrs)}")

    RT1 = cam1["RT"]
    K1 = deepcopy(cam1["K"])
    K1[0] *= W / im1_W
    K1[1] *= H / im1_H
    P1 = K1 @ RT1

    RT2 = cam2["RT"]
    K2 = deepcopy(cam2["K"])
    K2[0] *= W / im2_W
    K2[1] *= H / im2_H
    P2 = K2 @ RT2

    P1 = torch.from_numpy(P1).to(device, non_blocking=True)
    P2 = torch.from_numpy(P2).to(device, non_blocking=True)

    img1_corrs_homo = torch.cat((img1_corrs, torch.ones((len(img1_corrs), 1), device=device)), axis=1)
    img2_corrs_homo = torch.cat((img2_corrs, torch.ones((len(img2_corrs), 1), device=device)), axis=1)
    point = triangulate_points(P1, P2, img1_corrs_homo, img2_corrs_homo)
    point = point[:, :3]

    return point, colors


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", default=r"/mnt/selfcap/selfcap/0512_bike", type=str)
    parser.add_argument("--image_dir", default="images", type=str)
    parser.add_argument("--intri", default="intri.yml")
    parser.add_argument("--extri", default="extri.yml")
    parser.add_argument("--output_dir", default="pcds_roma")
    parser.add_argument("--n_points", default=10000, type=int)
    parser.add_argument("--mask_dir", default="", type=str)
    parser.add_argument("--bounds", default=[[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]], type=parse_list)
    parser.add_argument("--pairs", default=(("02", "04"), ("06", "08"), ("10", "12")), type=parse_list)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=-1, type=int)
    parser.add_argument("--force", default=False, action="store_true")

    # RoMA related config
    parser.add_argument("--certainty_thresh", default=0.2, type=float)
    parser.add_argument("--model_type", default="outdoor", type=str, choices=["indoor", "outdoor", "tiny", "v2"])

    # Performance optimization config
    parser.add_argument(
        "--prefetch_workers", default=4, type=int, help="Number of worker threads for image prefetching"
    )
    parser.add_argument("--cache_size", default=100, type=int, help="Maximum number of images to cache")
    parser.add_argument("--file_workers", default=2, type=int, help="Number of worker threads for ply file writing")

    args, _ = parser.parse_known_args()

    if args.bounds is not None:
        args.bounds = torch.as_tensor(args.bounds).to(device, non_blocking=True)

    # Create model
    print(f"Using RoMa model: {args.model_type}")
    if args.model_type == "v2":
        roma_model = RoMaV2()
        # we don't need that high precision here
        roma_model.H_hr = None
        roma_model.W_hr = None
        roma_model.bidirectional = False
        H, W = roma_model.H_lr, roma_model.W_lr
    else:
        roma_model = model_dict[args.model_type](device=device)
        H, W = roma_model.get_output_resolution() if args.model_type != "tiny" else (864, 864)

    data_root = args.data_root
    image_dir = join(data_root, args.image_dir)
    intri = join(data_root, args.intri)
    extri = join(data_root, args.extri)
    output_dir = args.output_dir if isabs(args.output_dir) else join(data_root, args.output_dir)
    if args.mask_dir:
        mask_dir = join(data_root, args.mask_dir)
        if not isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory {mask_dir} does not exist.")
    else:
        mask_dir = None
    os.makedirs(output_dir, exist_ok=True)

    cams = read_camera(intri, extri)
    cam_names = sorted(cams.keys())
    frames = os.listdir(join(image_dir, cam_names[0]))
    frames = sorted([x for x in frames])
    if args.end_idx == -1:
        frames = frames[args.start_idx :]
    else:
        frames = frames[args.start_idx : args.end_idx]

    view_list = args.pairs

    # Initialize image cache with prefetch workers
    image_cache = SequentialImageCache(max_size=args.cache_size, max_workers=args.prefetch_workers)
    target_size = (W, H)

    # Construct the actual access sequence to track when each image will be used
    access_sequence = construct_access_sequence(frames, view_list, image_dir)
    print(f"Constructed access sequence with {len(access_sequence)} accesses")

    # Add access sequence to cache for intelligent prefetching and eviction
    image_cache.add_paths(access_sequence, target_size)

    # Initialize async task executor
    async_task_executor = AsyncTaskExecutor(max_workers=args.file_workers)

    for i, f in enumerate(tqdm(frames, desc="Triangulating view")):
        frame_id = f.split(".")[0]
        pcd_path = join(output_dir, f"{frame_id}.ply")
        if isfile(pcd_path) and not args.force:
            print(f"skipping {pcd_path}")
            continue

        frame_pts = []
        frame_colors = []
        for view1, view2 in view_list:
            im1_path = join(image_dir, view1, f)
            im2_path = join(image_dir, view2, f)
            im1, im1_W, im1_H = image_cache.get_image_or_load(im1_path)
            im2, im2_W, im2_H = image_cache.get_image_or_load(im2_path)

            point, colors = roma_triangulate(
                roma_model=roma_model,
                im1=im1,
                im2=im2,
                ori_size1=(im1_H, im1_W),
                ori_size2=(im2_H, im2_W),
                H=H,
                W=W,
                cam1=cams[view1],
                cam2=cams[view2],
                model_type=args.model_type,
                mask_dir=mask_dir,
                certainty_thresh=args.certainty_thresh,
                frame_name=f,
                device=device,
            )
            frame_pts.append(point)
            frame_colors.append(colors)

        points = torch.cat(frame_pts, dim=0)
        colors = torch.cat(frame_colors, dim=0)

        if args.bounds is not None:
            mask = (points >= args.bounds[0]) & (points <= args.bounds[1])
            mask = mask.all(dim=-1).nonzero()[:, 0]
            points = points[mask]
            colors = colors[mask]

        if args.n_points > 0 and len(points) > args.n_points:
            indices = torch.randperm(len(points))[: args.n_points]
            points = points[indices]
            colors = colors[indices]

        async_task_executor.submit_task(export_pts, pcd_path, points, color=colors)

    image_cache.shutdown()

    # Wait for all async file writes to complete
    pending_count = async_task_executor.get_pending_count()
    if pending_count > 0:
        print(f"Waiting for {pending_count} pending file writes to complete...")
    async_task_executor.shutdown()
