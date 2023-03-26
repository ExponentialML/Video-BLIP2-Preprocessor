import torch
import json
import os
import random 
import numpy as np
import argparse

from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class PreProcessVideos:
    def __init__(
            self, 
            config_name,
            config_save_name,
            video_directory,
            limit,
            random_start_frame,
            beam_amount,
            prompt_amount,
            min_prompt_length,
            max_prompt_length,
            save_dir
        ):

        # Paramaters for parsing videos
        self.prompt_amount = prompt_amount
        self.video_directory = video_directory
        self.limit = limit
        self.random_start_frame = random_start_frame
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")

        # Parameters for BLIP2
        self.processor = None
        self.blip_model = None
        self.beam_amount = beam_amount
        self.min_length = min_prompt_length
        self.max_length = max_prompt_length

        # Helper parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.limit_msg = False
        self.save_dir = save_dir

        # Config parameters
        self.config_name = config_name
        self.config_save_name = config_save_name

    # Base dict to hold all the data.
    # {base_config}
    def build_base_config(self):
        return {
            "name": self.config_name,
            "data": []
        }

    # Video dict for individual videos.
    # {base_config: data -> [{video_path, num_frames, data}]}
    def build_video_config(self, video_path: str, num_frames: int):
        return {
            "video_path": video_path,
            "num_frames": num_frames,
            "data": []
        }

    # Dict for video frames and prompts / captions.
    # Gets the frame index, then gets a caption for the that frame and stores it.
    # {base_config: data -> [{name, num_frames, data: {frame_index, prompt}}]}
    def build_video_data(self, frame_index: int, prompt: str):
        return {
            "frame_index": frame_index,
            "prompt": prompt
        }

    # Load BLIP2 for processing
    def load_blip(self):
        print("Loading BLIP2")

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        model.to(self.device)

        self.processor = processor
        self.blip_model = model

    # Process the frames to get the length and image.
    # The limit parameter ensures we don't get near the max frame length.
    def video_processor(
            self, 
            video_reader: VideoReader, 
            num_frames: int, 
            limit: float, 
            random_start_frame=True
        ):
        limit_fix = 0.75
        random.seed()

        if limit > 1:
            if not self.limit_msg:
                print(f"Limit parameter cannot be greater than 1. setting to {limit_fix}")
                self.limit_msg = True
            limit = limit_fix

        frame_number = random.randrange(0, int(num_frames * limit)) if random_start_frame else 0
        frame_to_np = video_reader[frame_number].asnumpy()

        frame = torch.from_numpy(frame_to_np).permute(2,0,1)
        image = transforms.ToPILImage()(frame).convert("RGB")

        return frame_number, image

    def process_blip(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(
                **inputs, 
                num_beams=self.beam_amount, 
                min_length=self.min_length, 
                max_length=self.max_length
            )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True)[0].strip()
        
        return generated_text
    

    def save_train_config(self, config: dict):
        os.makedirs(self.save_dir, exist_ok=True)

        save_json = json.dumps(config, indent=4)
        with open(f"{self.save_dir}/{self.config_save_name}.json", 'w') as f:
            f.write(save_json)

    # Main loop for processing all videos.
    def process_videos(self):
        self.load_blip()
        config = self.build_base_config()

        if not os.path.exists(self.video_directory):
            raise ValueError(f"{self.video_directory} does not exist.")

        for _, _, files in tqdm(
                os.walk(self.video_directory), 
                desc=f"Processing videos in {self.video_directory}"
            ):
            for video in files:
                if video.endswith(self.vid_types):
                    video_path = f"{self.video_directory}/{video}"
                    video_reader = None

                    try:
                        video_reader = VideoReader(video_path, ctx=cpu(0))                
                    except:
                        print(f"Error loading {video_path}. Video may be unsupported or corrupt.")
                        continue
                    
                    # Another try catch block because decord isn't perfect.
                    try:
                        num_frames = int(len(video_reader))
                        video_config = self.build_video_config(video_path, num_frames)

                        # Secondary loop that process a specified amount of prompts, selects a random frame, then appends it.
                        for i in tqdm(
                                range(self.prompt_amount), 
                                desc=f"Processing {os.path.basename(video_path)}"
                            ):
                            frame_number, image = self.video_processor(
                                video_reader, 
                                num_frames, 
                                self.limit,
                                self.random_start_frame
                                )
                            prompt = self.process_blip(image)
                            video_data = self.build_video_data(frame_number, prompt)
                            video_config["data"].append(video_data)
                        
                        config['data'].append(video_config)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    continue

        print(f"Done. Saving train config to {self.save_dir}.")
        self.save_train_config(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_name', help="The name of the configuration.", type=str, default='My Config')
    parser.add_argument('--config_save_name', help="The name of the config file that's saved.", type=str, default='my_config')
    parser.add_argument('--video_directory', help="The directory where your videos are located.", type=str, default='./videos')
    parser.add_argument('--limit', help="The limit for the amount of frames to be read (num_frames * ~0-0.99)", type=float, default=0.85)
    parser.add_argument(
        '--random_start_frame', 
        help="Use random start frame when processing videos. Good for long videos where frames have different scenes and meanings.", 
        action='store_true', 
        default=True
    )
    parser.add_argument('--beam_amount', help="Amount for BLIP beam search.", type=int, default=7)
    parser.add_argument('--prompt_amount', help="The amount of prompts per video that is processed.", type=int, default=25)
    parser.add_argument('--min_prompt_length', help="Minimum words required in prompt.", type=int, default=15)
    parser.add_argument('--max_prompt_length', help="Maximum words required in prompt.", type=int, default=30)
    parser.add_argument('--save_dir', help="The directory to save the config to.", type=str, default=f"{os.getcwd()}/train_data")

    args = parser.parse_args()

    
    processor = PreProcessVideos(**vars(args))
    processor.process_videos()
