import os
import json
import argparse
import torch
from tqdm import tqdm
from easyvolcap.utils.console_utils import *
from diffsynth.data.mvdata import MultiVideoDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Flux Dataset Tester")
    parser.add_argument("--dataset_base_path", type=str, required=True, help="Base path for the dataset")
    parser.add_argument("--dataset_metadata_path", type=str, required=True, help="Path to the dataset metadata file")
    parser.add_argument("--data_file_keys", type=str, default="images", help="Comma-separated keys for data files to load")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per image")
    parser.add_argument("--height", type=int, default=256, help="Height to resize images to")
    parser.add_argument("--width", type=int, default=256, help="Width to resize images to")
    return parser

def main():
    # parse args
    parser = parse_args()
    args = parser.parse_args()

    # create dataset from metadata
    dataset = MultiVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=MultiVideoDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=8)
    for data in tqdm(dataloader):
        breakpoint()
        print(data)


if __name__ == '__main__':
    main()
