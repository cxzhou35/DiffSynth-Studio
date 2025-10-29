#Ryan Burgert 2024

#Setup:
#    Run this in a Jupyter Notebook on a computer with at least one GPU
#        `sudo apt install ffmpeg git`
#        `pip install rp`
#    The first time you run this it might be a bit slow (it will download necessary models)
#    The `rp` package will take care of installing the rest of the python packages for you

import rp
import os
import json
import argparse
import numpy as np
from glob import glob
from itertools import product
from typing import List
from diffsynth.utils.noise_util import get_noise_from_video

def main(image_paths: List[str], output_dir:str, image_height:int, image_width:int):
    """
    Takes a video URL or filepath and an output folder path
    It then resizes that video to height=480, width=720, 49 frames (CogVidX's dimensions)
    Then it calculates warped noise at latent resolution (i.e. 1/8 of the width and height) with 16 channels
    It saves that warped noise, optical flows, and related preview videos and images to the output folder
    The main file you need is <output_folder>/noises.npy which is the gaussian noises in (H,W,C) form
    """

    # if rp.folder_exists(output_dir):
    #     raise RuntimeError(f"The given output_folder={repr(output_dir)} already exists! To avoid clobbering what might be in there, please specify a folder that doesn't exist so I can create one for you. Alternatively, you could delete that folder if you don't care whats in it.")

    FRAME = 2**-1 #We immediately resize the input frames by this factor, before calculating optical flow
                  #The flow is calulated at (input size) × FRAME resolution.
                  #Higher FLOW values result in slower optical flow calculation and higher intermediate noise resolution
                  #Larger is not always better - watch the preview in Jupyter to see if it looks good!

    FLOW = 2**3   #Then, we use bilinear interpolation to upscale the flow by this factor
                  #We warp the noise at (input size) × FRAME × FLOW resolution
                  #The noise is then downsampled back to (input size)
                  #Higher FLOW values result in more temporally consistent noise warping at the cost of higher VRAM usage and slower inference time
    LATENT = 8    #We further downsample the outputs by this amount - because 8 pixels wide corresponds to one latent wide in Stable Diffusion
                  #The final output size is (input size) ÷ LATENT regardless of FRAME and FLOW

    #LATENT = 1    #Uncomment this line for a prettier visualization! But for latent diffusion models, use LATENT=8

    #You can also use video files or URLs
    # video = "https://www.shutterstock.com/shutterstock/videos/1100085499/preview/stock-footage-bremen-germany-october-old-style-carousel-moving-on-square-in-city-horses-on-traditional.webm"

    frames = []
    for image_path in image_paths:
        frames.append(rp.load_image(image_path))
    video = np.stack(frames)
    breakpoint()

    #Preprocess the video
    video = rp.resize_list(video, length=len(video)) #Stretch or squash video to 49 frames (CogVideoX's length)
    video = rp.resize_images_to_hold(video, height=image_height, width=image_width)
    video = rp.crop_images(video, height=image_height, width=image_width, origin='center') #Make the resolution 480x720 (CogVideoX's resolution)
    video = rp.as_numpy_array(video)


    #See this function's docstring for more information!
    output = get_noise_from_video(
        video,
        remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,         #Set this to False if you just want the noises without saving to a numpy file

        noise_channels=16,
        output_folder=output_dir,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor=round(FRAME * FLOW) * LATENT,
    )

    output.first_frame_path = rp.save_image(video[0],rp.path_join(output_dir,'first_frame.png'))

    # rp.save_video_mp4(video, rp.path_join(output_folder, 'input.mp4'), framerate=12, video_bitrate='max')

    #output.numpy_noises_downsampled = as_numpy_images(
        #nw.resize_noise(
            #as_torch_images(x),
            #1 / 8,
        #)for x
    #)
    #
    #output.numpy_noises_downsampled_path = path_join(output_folder, 'noises_downsampled.npy')
    #np.save(numpy_noises_downsampled_path, output.numpy_noises_downsampled)

    print("Noise shape:"  , output.numpy_noises.shape)
    print("Flow shape:"   , output.numpy_flows.shape)
    print("Output folder:", output.output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("-i", "--data_root", type=str, help="Base path of the dataset.")
    parser.add_argument("-o", "--output_dir", type=str, default="output/noise_warping", help="Path to output")
    parser.add_argument("--height", type=int, default=1080, help="Height of the frame.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the frame.")
    args = parser.parse_args()

    image_files = sorted(os.listdir(args.data_root))
    image_paths = [os.path.join(args.data_root, f) for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    main(image_paths, args.output_dir, args.height, args.width)
