import torch
import torchvision
import imageio
import os
import json
import random
import pandas
import imageio.v3 as iio
from PIL import Image
from .utils import (
    LoadTorchPickle,
    RouteByType,
    ToAbsolutePath,
    LoadImage,
    ImageCropAndResize,
    RouteByExtensionName,
    LoadGIF,
    LoadVideo,
    ToList,
    SequencialProcess,
)


class MultiVideoDataset(torch.utils.data.Dataset):
    """
    Multi video dataset
    """
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        use_temporal_sample=False,
        temporal_window_size=4,
        use_spatial_sample=False,
        spatial_window_size=4,
        sample_mode="fixed",
        mixed_sampling_probs=(0.4, 0.4, 0.2),
        joint_temporal_window_size=2,
        joint_spatial_window_size=2,
        use_tem_key_frame=False,
        key_frame_chunk=None,
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.temporal_window_size = temporal_window_size if use_temporal_sample else 1
        self.spatial_window_size = spatial_window_size if use_spatial_sample else 1
        self.sample_mode = sample_mode
        self.mixed_sampling_probs = mixed_sampling_probs
        self.joint_temporal_window_size = joint_temporal_window_size
        self.joint_spatial_window_size = joint_spatial_window_size
        self.use_tem_key_frame = use_tem_key_frame
        self.key_frame_chunk = key_frame_chunk if use_tem_key_frame else 1
        self.scene_infos = {}
        self.load_metadata(metadata_path)
        self.parse_metadata()

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                ((("jpg", "jpeg", "png", "webp")), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                ((("gif",)), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                ((("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, "r") as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _infer_scene_id(self, item):
        if "scene" in item and item["scene"] not in (None, ""):
            return str(item["scene"])
        image_path = item.get("image", "")
        parts = image_path.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == "data":
            return parts[1]
        if "data" in parts:
            idx = parts.index("data")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "default"

    def _select_window(self, values, center_idx, window_size):
        if window_size <= 1 or len(values) <= 1:
            return [values[center_idx]]
        if len(values) <= window_size:
            return list(values)
        if center_idx + 1 < window_size:
            return list(values[:window_size])
        if len(values) - center_idx < window_size:
            return list(values[-window_size:])
        start = center_idx - window_size // 2
        end = start + window_size
        return list(values[start:end])

    def _resolve_scene_item(self, data_id):
        index = data_id % len(self.data)
        item = self.data[index]
        scene_id = item["__scene_id"]
        cam_id = item["__cam_id"]
        frame_id = item["__frame_id"]
        scene_info = self.scene_infos[scene_id]
        return index, item, scene_id, cam_id, frame_id, scene_info

    def parse_metadata(self):
        if self.load_from_cache or len(self.data) == 0:
            return

        self.scene_infos = {}
        self.scene_ids = []

        for index, item in enumerate(self.data):
            scene_id = self._infer_scene_id(item)
            cam_id = int(item["view_id"])
            frame_id = int(item["frame_id"])
            item["__scene_id"] = scene_id
            item["__cam_id"] = cam_id
            item["__frame_id"] = frame_id

            if scene_id not in self.scene_infos:
                self.scene_infos[scene_id] = {
                    "cam_ids": set(),
                    "frame_ids": set(),
                    "index_map": {},
                }
                self.scene_ids.append(scene_id)

            scene_info = self.scene_infos[scene_id]
            scene_info["cam_ids"].add(cam_id)
            scene_info["frame_ids"].add(frame_id)
            scene_info["index_map"][(cam_id, frame_id)] = index

        for scene_id, scene_info in self.scene_infos.items():
            scene_info["cam_ids"] = sorted(scene_info["cam_ids"])
            scene_info["frame_ids"] = sorted(scene_info["frame_ids"])
            scene_info["cam_id_to_idx"] = {c: i for i, c in enumerate(scene_info["cam_ids"])}
            scene_info["frame_id_to_idx"] = {f: i for i, f in enumerate(scene_info["frame_ids"])}
            scene_info["frame_id_set"] = set(scene_info["frame_ids"])

        all_cam_ids = sorted({item["__cam_id"] for item in self.data})
        all_frame_ids = sorted({item["__frame_id"] for item in self.data})
        self.cam_ids = all_cam_ids
        self.frame_ids = all_frame_ids
        self.n_cams = len(all_cam_ids)
        self.n_frames = len(all_frame_ids)
        self._cam_id_to_idx = {c: i for i, c in enumerate(all_cam_ids)}

    def get_mvdata_ids(self, data_id, domain="temporal"):
        _, _, scene_id, cam_id, frame_id, scene_info = self._resolve_scene_item(data_id)

        if domain == "temporal":
            frame_idx = scene_info["frame_id_to_idx"][frame_id]
            temporal_ids = self._select_window(scene_info["frame_ids"], frame_idx, self.temporal_window_size)
            return [scene_info["index_map"][(cam_id, fid)] for fid in temporal_ids if (cam_id, fid) in scene_info["index_map"]]

        if domain == "spatial":
            cam_idx = scene_info["cam_id_to_idx"][cam_id]
            spatial_cam_ids = self._select_window(scene_info["cam_ids"], cam_idx, self.spatial_window_size)
            return [scene_info["index_map"][(cid, frame_id)] for cid in spatial_cam_ids if (cid, frame_id) in scene_info["index_map"]]

        raise ValueError(f"Unknown domain: {domain}")

    def get_joint_data_ids(self, data_id):
        _, _, scene_id, cam_id, frame_id, scene_info = self._resolve_scene_item(data_id)
        frame_idx = scene_info["frame_id_to_idx"][frame_id]
        cam_idx = scene_info["cam_id_to_idx"][cam_id]
        temporal_ids = self._select_window(scene_info["frame_ids"], frame_idx, self.joint_temporal_window_size)
        spatial_cam_ids = self._select_window(scene_info["cam_ids"], cam_idx, self.joint_spatial_window_size)
        data_ids = []
        for fid in temporal_ids:
            for cid in spatial_cam_ids:
                if (cid, fid) in scene_info["index_map"]:
                    data_ids.append(scene_info["index_map"][(cid, fid)])
        return data_ids

    def _normalize_mixed_mode_probs(self):
        probs = self.mixed_sampling_probs
        if isinstance(probs, str):
            probs = [float(x) for x in probs.split(",")]
        probs = list(probs)
        if len(probs) != 3:
            raise ValueError(f"mixed_sampling_probs must have 3 values, got {probs}")
        modes = []
        weights = []
        temporal_valid = self.temporal_window_size > 1
        spatial_valid = self.spatial_window_size > 1
        joint_valid = self.joint_temporal_window_size > 1 and self.joint_spatial_window_size > 1
        for mode, weight, valid in [
            ("temporal", probs[0], temporal_valid),
            ("spatial", probs[1], spatial_valid),
            ("joint", probs[2], joint_valid),
        ]:
            if valid and weight > 0:
                modes.append(mode)
                weights.append(float(weight))
        if not modes:
            return ["single"], [1.0]
        return modes, weights

    def get_mixed_data_ids(self, data_id):
        modes, weights = self._normalize_mixed_mode_probs()
        mode = random.choices(modes, weights=weights, k=1)[0]
        if mode == "temporal":
            return self.get_mvdata_ids(data_id, domain="temporal")
        if mode == "spatial":
            return self.get_mvdata_ids(data_id, domain="spatial")
        if mode == "joint":
            return self.get_joint_data_ids(data_id)
        return [data_id % len(self.data)]

    def get_data_ids(self, data_id):
        if self.sample_mode == "mixed":
            return self.get_mixed_data_ids(data_id)
        if self.temporal_window_size > 1 and self.spatial_window_size > 1:
            raise NotImplementedError("Simultaneous temporal and spatial sampling is not supported in fixed mode. Use sample_mode=\"mixed\" instead.")
        if self.temporal_window_size > 1:
            return self.get_mvdata_ids(data_id, domain="temporal")
        if self.spatial_window_size > 1:
            return self.get_mvdata_ids(data_id, domain="spatial")
        return [data_id % len(self.data)]

    def _resolve_key_frame_id(self, frame_ids, target_frame_id):
        if target_frame_id in frame_ids:
            return target_frame_id
        previous = [fid for fid in frame_ids if fid <= target_frame_id]
        if previous:
            return previous[-1]
        return frame_ids[0]

    def get_controlnet_key_frame(self, data_id, key_data_key="controlnet_images"):
        _, _, scene_id, cam_id, frame_id, scene_info = self._resolve_scene_item(data_id)
        key_frame_id = (frame_id // self.key_frame_chunk) * self.key_frame_chunk
        min_frame_id = scene_info["frame_ids"][0]
        if key_frame_id > min_frame_id and frame_id % self.key_frame_chunk == 0:
            key_frame_id -= 1
        key_frame_id = self._resolve_key_frame_id(scene_info["frame_id_set"], key_frame_id)
        key_index = scene_info["index_map"].get((cam_id, key_frame_id))
        if key_index is None:
            raise KeyError(f"Missing key frame for scene={scene_id}, cam={cam_id}, frame={key_frame_id}")
        return self.data[key_index][key_data_key]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
            return data
        else:
            data_ids = self.get_data_ids(data_id)
            datas = []
            for id in data_ids:
                data = self.data[id].copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            data[key] = self.main_data_operator(data[key])
                if self.use_tem_key_frame:
                    data["controlnet_key_images"] = self.main_data_operator(
                        self.get_controlnet_key_frame(id, key_data_key="controlnet_images")
                    )
                datas.append(data)
            return datas

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
