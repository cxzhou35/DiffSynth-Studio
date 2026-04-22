import os
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


def construct_cond_data(image_dir: str, cond_image_dir: str, prompts: list[str], metafile_path: str, cond_type: str, remove_prefix: str, frame_range: list[int], view_range: list[int]):
    # check input and cond image directory format
    image_dir_format = check_dir_format(image_dir)
    cond_image_dir_format = check_dir_format(cond_image_dir)
    assert image_dir_format == cond_image_dir_format, f"Input image directory format and condition image directory format must be the same"

    metafile_handler = FileHandler(metafile_path)
    metafile_type = metafile_handler.file_type
    total_pair_num = 0

    s, e, t = frame_range
    vs, ve, vt = view_range

    # construct condition data and write to metadata file
    # NOTE: data format: evc image_dir/view_dir/image
    if image_dir_format == "evc":
        view_dirs = sorted(os.listdir(cond_image_dir))[vs:ve+1:vt]
        log(f"Found {len(view_dirs)} views in {cond_image_dir}")
        if metafile_type == "csv":
            metafile_handler.update_data_container(["image", "prompt", f"{cond_type}_images", "view_id", "frame_id"])
        for idx, view_dir in enumerate(tqdm(view_dirs, desc=f"Constructing condition {cond_type} data")):
            image_view_dir = os.path.join(image_dir, view_dir)
            cond_image_view_dir = os.path.join(cond_image_dir, view_dir)
            image_list = sorted(os.listdir(image_view_dir))[s:e+1:t]
            cond_image_list = sorted(os.listdir(cond_image_view_dir))[s:e+1:t]
            for idx_ in range(len(image_list)):
                image_path = os.path.join(image_view_dir, image_list[idx_])
                cond_image_path = os.path.join(cond_image_view_dir, cond_image_list[idx_])
                if metafile_type == "csv":
                    data = [
                        image_path.replace(remove_prefix, ""),
                        prompts[idx] if len(prompts) > 1 else prompts[0],
                        cond_image_path.replace(remove_prefix, ""),
                        f"{view_dir}",
                        # "frame_id": f"{image_list[idx_].split('.')[0]}"
                        f"{idx_:02d}"
                    ]
                elif metafile_type in ["json", "jsonl"]:
                    data = {
                        "image": image_path.replace(remove_prefix, ""),
                        "prompt": prompts[idx] if len(prompts) > 1 else prompts[0],
                        f"{cond_type}_images": cond_image_path.replace(remove_prefix, ""),
                        "view_id": f"{view_dir}",
                        # "frame_id": f"{image_list[idx_].split('.')[0]}"
                        "frame_id": f"{idx_:02d}"
                    }
                else:
                    raise ValueError(f"Unsupported metafile type: {metafile_type}")
                metafile_handler.update_data_container(data)
            total_pair_num += len(image_list)
        metafile_handler.write(metafile_handler.data_container)

    # NOTE: data format: general image_dir/image
    else:
        image_list = sorted(os.listdir(image_dir))
        cond_image_list = sorted(os.listdir(cond_image_dir))
        total_pair_num += len(image_list)
        for idx in tqdm(range(len(image_list)), desc=f"Constructing condition {cond_type} data"):
            image_path = os.path.join(image_dir, image_list[idx])
            cond_image_path = os.path.join(cond_image_dir, cond_image_list[idx])
            data = [image_path.replace(remove_prefix, ""), prompts[0], cond_image_path.replace(remove_prefix, "")]
            metafile_handler.update_data_container(data)
        metafile_handler.write(metafile_handler.data_container)

    return total_pair_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("-c", "--cond_image_dir", type=str, required=True, help="Input condition image directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-p", "--prompt", type=str, default="test prompt", help="Input prompt")
    parser.add_argument("-pd", "--prompt_dir", type=str, default="prompts", help="Input prompt directory")
    parser.add_argument("-mt", "--meta_type", type=str, default="json", help="File type of metadata")
    parser.add_argument("-ct", "--cond_type", type=str, default="controlnet", help="Condition type (etc. controlnet, kontext)")
    parser.add_argument("-s", "--split", type=str, default="train", help="Dataset split (train, test, eval)")
    parser.add_argument("-rp", "--remove_prefix", type=str, default="data/old_tim_1440p_120f/", help="Remove prefix in path")
    parser.add_argument("-fr", "--frame_range", nargs=3, type=int, help="Frame ranges for data selecting")
    parser.add_argument("-vr", "--view_range", nargs=3, type=int, help="View ranges for data selecting")

    parse_args = parser.parse_args()
    return parse_args


def main():
    args = parse_args()
    image_dir = args.image_dir
    cond_image_dir = args.cond_image_dir
    output_dir = args.output_dir
    prompt = args.prompt
    prompt_dir = args.prompt_dir
    meta_type = args.meta_type
    cond_type = args.cond_type
    split = args.split
    remove_prefix = args.remove_prefix
    frame_range = args.frame_range
    view_range = args.view_range

    # if prompt_dir is not empty, read prompt from prompt_dir
    if prompt_dir:
        prompts = []
        prompt_files = sorted(os.listdir(prompt_dir))
        for prompt_file in prompt_files:
            with open(os.path.join(prompt_dir, prompt_file), "r") as f:
                prompts.append(f.read())
    else:
        prompts = [prompt]

    if frame_range is None:
        view_dir_tmp = os.listdir(image_dir)[0]
        frame_range = [0, len(os.listdir(os.path.join(image_dir, view_dir_tmp)))-1, 1]
    if view_range is None:
        view_range = [0, len(os.listdir(image_dir))-1, 1]

    os.makedirs(output_dir, exist_ok=True)
    metafile_path = os.path.join(output_dir, f"metadata_{split}.{meta_type}")
    total_pair_num = construct_cond_data(image_dir, cond_image_dir, prompts, metafile_path, cond_type, remove_prefix, frame_range, view_range)
    log(f"Constructed {total_pair_num} {cond_type} condition data pairs.")
    log(f"Saved metadata to {metafile_path}")


if __name__ == "__main__":
    main()
