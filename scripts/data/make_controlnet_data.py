import os
import csv
import argparse
from tqdm import tqdm
from easyvolcap.utils.console_utils import log

BATCH_ITEMS=['image', 'prompt', 'controlnet_image']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("-c", "--controlnet_image_dir", type=str, required=True, help="Input controlnet image directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-p", "--prompt", type=str, default="test prompt", help="Input prompt")

    parse_args = parser.parse_args()
    return parse_args

def check_dir_format(image_dir: str) -> str:
    tmp_item = os.listdir(image_dir)[0]
    if os.path.isdir(os.path.join(image_dir, tmp_item)):
        return "evc"
    elif os.path.isfile(os.path.join(image_dir, tmp_item)):
        return "general"
    else:
        raise ValueError(f"Invalid image directory format: {image_dir}")

def construct_controlnet_data(image_dir: str, controlnet_image_dir: str, output_dir: str, prompt: str):
    # check input and controlnet image directory format
    input_image_dir_format = check_dir_format(image_dir)
    controlnet_image_dir_format = check_dir_format(controlnet_image_dir)
    assert input_image_dir_format == controlnet_image_dir_format, f"Input image directory format and controlnet image directory format must be the same"

    output_csv_path = os.path.join(output_dir, "metadata.csv")

    # create output metadata file
    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(BATCH_ITEMS)

    total_pair_num = 0

    # construct controlnet data and write to metadata file
    # NOTE: data format: evc image_dir/view_dir/image
    if input_image_dir_format == "evc":
        view_dirs = sorted(os.listdir(image_dir))
        log(f"Found {len(view_dirs)} views in {image_dir}")
        for view_dir in view_dirs:
            input_image_view_dir = os.path.join(image_dir, view_dir)
            controlnet_image_view_dir = os.path.join(controlnet_image_dir, view_dir)
            input_image_list = sorted(os.listdir(input_image_view_dir))
            input_controlnet_image_list = sorted(os.listdir(controlnet_image_view_dir))
            for input_image, controlnet_image in tqdm(zip(input_image_list, input_controlnet_image_list), desc="Constructing controlnet data"):
                input_image_path = os.path.join(input_image_view_dir, input_image)
                controlnet_image_path = os.path.join(controlnet_image_view_dir, controlnet_image)
                with open(output_csv_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([input_image_path, prompt, controlnet_image_path])
            total_pair_num += len(input_image_list)

    # NOTE: data format: general image_dir/image
    else:
        input_image_list = sorted(os.listdir(image_dir))
        controlnet_image_list = sorted(os.listdir(controlnet_image_dir))
        total_pair_num += len(input_image_list)
        for input_image, controlnet_image in tqdm(zip(input_image_list, controlnet_image_list), desc="Constructing controlnet data"):
            input_image_path = os.path.join(image_dir, input_image)
            controlnet_image_path = os.path.join(controlnet_image_dir, controlnet_image)
            with open(output_csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([input_image_path, prompt, controlnet_image_path])

    return output_csv_path, total_pair_num



def main():
    args = parse_args()
    image_dir = args.image_dir
    controlnet_image_dir = args.controlnet_image_dir
    output_dir = args.output_dir
    prompt = args.prompt

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path, total_pair_num = construct_controlnet_data(image_dir, controlnet_image_dir, output_dir, prompt)
    log(f"Constructed {total_pair_num} controlnet data pairs.")
    log(f"Saved the metadata file to {output_csv_path}")


if __name__ == "__main__":
    main()
