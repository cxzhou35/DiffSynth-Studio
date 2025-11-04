import torch
import torchvision
import imageio
import os
import json
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
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
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
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def parse_metadata(self):
        extract_cam_id = lambda x: int(x['view_id'])
        extract_frame_id = lambda x: int(x['frame_id'])
        max_cam_id, min_cam_id = max(extract_cam_id(x) for x in self.data), min(extract_cam_id(x) for x in self.data)
        self.cam_ids = list(range(min_cam_id, max_cam_id + 1))
        self.n_cams = len(self.cam_ids)
        max_frame_id, min_frame_id = max(extract_frame_id(x) for x in self.data), min(extract_frame_id(x) for x in self.data)
        self.frame_ids = list(range(min_frame_id, max_frame_id + 1))
        self.n_frames = len(self.frame_ids)

    def get_mvdata_ids(self, data_id, domain='temporal'):
        data_id = self.data[data_id % len(self.data)]
        cam_id = int(data_id['view_id'])
        frame_id = int(data_id['frame_id'])
        data_ids = []

        if domain == 'temporal':
            # Temporal sampling
            if frame_id-self.frame_ids[0]+1 < self.temporal_window_size:
                temporal_ids = list(range(self.frame_ids[0], self.frame_ids[0]+self.temporal_window_size))
            elif self.frame_ids[-1]-frame_id+1 < self.temporal_window_size:
                temporal_ids = list(range(self.frame_ids[-1]-self.temporal_window_size+1, self.frame_ids[-1]+1))
            else:
                temporal_ids = list(range(frame_id-self.temporal_window_size//2, frame_id+(self.temporal_window_size+1)//2))
            data_ids = [self.n_frames*(cam_id-self.cam_ids[0]) + (fid-self.frame_ids[0]) for fid in temporal_ids]

        if domain == 'spatial':
            # Spatial sampling
            if cam_id-self.cam_ids[0]+1 < self.spatial_window_size:
                spatial_ids = list(range(self.cam_ids[0], self.cam_ids[0]+self.spatial_window_size))
            elif self.cam_ids[-1]-cam_id+1 < self.spatial_window_size:
                spatial_ids = list(range(self.cam_ids[-1]-self.spatial_window_size+1, self.cam_ids[-1]+1))
            else:
                spatial_ids = list(range(cam_id-self.spatial_window_size//2, cam_id+(self.spatial_window_size+1)//2))
            data_ids = [(sid-self.cam_ids[0])*self.n_frames + (frame_id-self.frame_ids[0]) for sid in spatial_ids]

        return data_ids

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
            return data
        else:
            data_ids = self.get_mvdata_ids(data_id)
            datas = []
            for id in data_ids:
                data = self.data[id].copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            data[key] = self.main_data_operator(data[key])
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
