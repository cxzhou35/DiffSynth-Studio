import os
import csv
import argparse
from tqdm import tqdm
from easyvolcap.utils.console_utils import log, tqdm
from utils import DotDict, FileHandler


def check_dir_format(image_dir: str) -> str:
    tmp_item = os.listdir(image_dir)[0]
    if os.path.isdir(os.path.join(image_dir, tmp_item)):
        return "evc"
    elif os.path.isfile(os.path.join(image_dir, tmp_item)):
        return "general"
    else:
        raise ValueError(f"Invalid image directory format: {image_dir}")


def construct_cond_data(image_dir: str, cond_image_dir: str, prompt: str, metafile_path: str, cond_type: str, remove_prefix):
    # check input and cond image directory format
    image_dir_format = check_dir_format(image_dir)
    cond_image_dir_format = check_dir_format(cond_image_dir)
    assert image_dir_format == cond_image_dir_format, f"Input image directory format and condition image directory format must be the same"

    metafile_handler = FileHandler(metafile_path)
    total_pair_num = 0

    # construct condition data and write to metadata file
    # NOTE: data format: evc image_dir/view_dir/image
    if image_dir_format == "evc":
        view_dirs = sorted(os.listdir(image_dir))
        log(f"Found {len(view_dirs)} views in {image_dir}")
        for view_dir in tqdm(view_dirs, desc=f"Constructing condition {cond_type} data"):
            image_view_dir = os.path.join(image_dir, view_dir)
            cond_image_view_dir = os.path.join(cond_image_dir, view_dir)
            image_list = sorted(os.listdir(image_view_dir))
            cond_image_list = sorted(os.listdir(cond_image_view_dir))
            for idx in range(len(image_list)):
                image_path = os.path.join(image_view_dir, image_list[idx])
                cond_image_path = os.path.join(cond_image_view_dir, cond_image_list[idx])
                # data = [input_image_path, prompt, cond_image_path]
                data = {
                    "image": image_path.replace(remove_prefix, ""),
                    "prompt": prompt,
                    f"{cond_type}_images": cond_image_path.replace(remove_prefix, ""),
                    "view_id": f"{view_dir}",
                    "frame_id": f"{image_list[idx].split('.')[0]}"
                }
                image_id = f"{view_dir}_{image_list[idx].split('.')[0]}"
                metafile_handler.update_data_container({image_id: data})
            total_pair_num += len(image_list)
        metafile_handler.write(metafile_handler.data_container)

    # NOTE: data format: general image_dir/image
    else:
        image_list = sorted(os.listdir(image_dir))
        cond_image_list = sorted(os.listdir(cond_image_dir))
        total_pair_num += len(image_list)
        for idx in tqdm(range(len(image_list)), desc=f"Constructing condition {cond_type} data"):
            image_path = os.path.join(image_view_dir, image_list[idx])
            cond_image_path = os.path.join(cond_image_view_dir, cond_image_list[idx])
            data = [image_path.replace(remove_prefix, ""), prompt, cond_image_path.replace(remove_prefix, "")]
            data_container = metafile_handler.update_data_container(data)
        metafile_handler.write(data_container)

    return total_pair_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("-c", "--cond_image_dir", type=str, required=True, help="Input condition image directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-p", "--prompt", type=str, default="test prompt", help="Input prompt")
    parser.add_argument("-mt", "--meta_type", type=str, default="json", help="File type of metadata")
    parser.add_argument("-ct", "--cond_type", type=str, default="controlnet", help="Condition type (etc. controlnet, kontext)")
    parser.add_argument("-rp", "--remove_prefix", type=str, default="/home/zhouchenxu/codes/DiffSynth-Studio/", help="Remove prefix in path")

    parse_args = parser.parse_args()
    return parse_args


def main():
    args = parse_args()
    image_dir = args.image_dir
    cond_image_dir = args.cond_image_dir
    output_dir = args.output_dir
    prompt = args.prompt
    meta_type = args.meta_type
    cond_type = args.cond_type
    remove_prefix = args.remove_prefix

    os.makedirs(output_dir, exist_ok=True)
    metafile_path = os.path.join(output_dir, f"metadata.{meta_type}")
    total_pair_num = construct_cond_data(image_dir, cond_image_dir, prompt, metafile_path, cond_type, remove_prefix)
    log(f"Constructed {total_pair_num} {cond_type} condition data pairs.")
    log(f"Saved metadata to {metafile_path}")


if __name__ == "__main__":
    main()
