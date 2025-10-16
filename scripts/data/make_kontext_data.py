import os
import argparse
import sys
import csv
from tqdm import tqdm

BATCH_ITEMS=['image', 'prompt', 'kontext_images']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("-c", "--kontext_image_dir", type=str, required=True, help="Input kontext image directory")
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

def construct_kontext_data(image_dir: str, kontext_image_dir: str, output_dir: str, prompt: str):
    # check input and kontext image directory format
    input_image_dir_format = check_dir_format(image_dir)
    kontext_image_dir_format = check_dir_format(kontext_image_dir)
    assert input_image_dir_format == kontext_image_dir_format, f"Input image directory format and kontext image directory format must be the same"

    output_csv_path = os.path.join(output_dir, "metadata_kontext_data.csv")

    # create output metadata file
    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(BATCH_ITEMS)

    total_pair_num = 0

    # construct kontext data and write to metadata file
    # NOTE: data format: evc image_dir/view_dir/image
    if input_image_dir_format == "evc":
        view_dirs = sorted(os.listdir(image_dir))
        print(f"Found {len(view_dirs)} views in {image_dir}")
        for view_dir in view_dirs:
            input_image_view_dir = os.path.join(image_dir, view_dir)
            kontext_image_view_dir = os.path.join(kontext_image_dir, view_dir)
            input_image_list = sorted(os.listdir(input_image_view_dir))
            input_kontext_image_list = sorted(os.listdir(kontext_image_view_dir))
            for input_image, kontext_image in tqdm(zip(input_image_list, input_kontext_image_list), desc="Constructing kontext data"):
                input_image_path = os.path.join(input_image_view_dir, input_image)
                kontext_image_path = os.path.join(kontext_image_view_dir, kontext_image)
                with open(output_csv_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([input_image_path, prompt, kontext_image_path])
            total_pair_num += len(input_image_list)

    # NOTE: data format: general image_dir/image
    else:
        input_image_list = sorted(os.listdir(image_dir))
        kontext_image_list = sorted(os.listdir(kontext_image_dir))
        total_pair_num += len(input_image_list)
        for input_image, kontext_image in tqdm(zip(input_image_list, kontext_image_list), desc="Constructing kontext data"):
            input_image_path = os.path.join(image_dir, input_image)
            kontext_image_path = os.path.join(kontext_image_dir, kontext_image)
            with open(output_csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([input_image_path, prompt, kontext_image_path])

    return output_csv_path, total_pair_num



def main():
    args = parse_args()
    image_dir = args.image_dir
    kontext_image_dir = args.kontext_image_dir
    output_dir = args.output_dir
    prompt = args.prompt

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path, total_pair_num = construct_kontext_data(image_dir, kontext_image_dir, output_dir, prompt)
    print(f"Constructed {total_pair_num} kontext data pairs.")
    print(f"Saved the metadata file to {output_csv_path}")


if __name__ == "__main__":
    main()
