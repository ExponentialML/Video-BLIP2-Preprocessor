import torch
import torchvision
import json
import os
import random 
import numpy as np
import argparse
import decord

from einops import rearrange
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration

decord.bridge.set_bridge('torch')

class PreProcessVideos:
    def __init__(
            self, 
            config_name,
            config_save_name,
            video_directory,
            random_start_frame,
            clip_frame_data,
            max_frames,
            beam_amount,
            prompt_amount,
            min_prompt_length,
            max_prompt_length,
            save_dir
        ):

        # Paramaters for parsing videos
        self.prompt_amount = prompt_amount
        self.video_directory = video_directory
        self.random_start_frame = random_start_frame
        self.clip_frame_data = clip_frame_data
        self.max_frames = max_frames
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")

        # Parameters for BLIP2
        self.processor = None
        self.blip_model = None
        self.beam_amount = beam_amount
        self.min_length = min_prompt_length
        self.max_length = max_prompt_length

        # Helper parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            random_start_frame=True,
            frame_num=0
        ):

        frame_number = (
            random.randrange(0, int(num_frames)) if random_start_frame else frame_num
            )
        frame = video_reader[frame_number].permute(2,0,1)
        image = transforms.ToPILImage()(frame).convert("RGB")
        return frame_number, image

    def get_frame_range(self, derterministic):
        return range(self.prompt_amount) if self.random_start_frame else derterministic

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
    
    def get_out_paths(self, prompt, frame_number):
        out_name= f"{prompt}_{str(frame_number)}"
        save_path = f"{self.save_dir}/{self.config_save_name}"
        save_filepath = f"{save_path}/{out_name}.mp4"

        return out_name, save_path, save_filepath

    def save_train_config(self, config: dict):
        os.makedirs(self.save_dir, exist_ok=True)

        save_json = json.dumps(config, indent=4)
        save_dir = f"{self.save_dir}/{self.config_save_name}"
        
        with open(f"{save_dir}.json", 'w') as f:
            f.write(save_json)

    def save_video(self, save_path, save_filepath, frames):
        os.makedirs(save_path, exist_ok=True)
        torchvision.io.write_video(save_filepath, frames, fps=30)

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
                    derterministic_range = None
                    video_len = 0
                    try:
                        video_reader = VideoReader(video_path, ctx=cpu(0))
                        video_len = len(video_reader)
                        frame_step = abs(video_len // self.prompt_amount)
                        derterministic_range = range(1, abs(video_len - 1), frame_step)          
                    except:
                        print(f"Error loading {video_path}. Video may be unsupported or corrupt.")
                        continue
                    
                    # Another try catch block because decord isn't perfect.
                    try:
                        num_frames = int(len(video_reader))
                        video_config = self.build_video_config(video_path, num_frames)

                        # Secondary loop that process a specified amount of prompts, selects a random frame, then appends it.
                        for i in tqdm(
                                self.get_frame_range(derterministic_range), 
                                desc=f"Processing {os.path.basename(video_path)}"
                            ):
                            frame_number, image = self.video_processor(
                                video_reader, 
                                num_frames, 
                                self.random_start_frame,
                                frame_num=i
                            )

                            prompt = self.process_blip(image)
                            video_data = self.build_video_data(frame_number, prompt)

                            if self.clip_frame_data:

                                # Minimum value, frame number, max value (length of entire video)
                                max_range = abs(len(video_reader) - 1)
                                frame_number = i
                                frame_number = sorted((1, frame_number, max_range))[1]

                                frame_range = range(frame_number, max_range)
                                frame_range_nums= list(frame_range)

                                frames = video_reader.get_batch(frame_range_nums[:self.max_frames])

                                out_name, save_path, save_filepath = self.get_out_paths(prompt, frame_number)
                                
                                self.save_video(save_path, save_filepath, frames)

                                video_data['clip_path'] = save_filepath
                                video_config["data"].append(video_data)

                            else:
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
    parser.add_argument(
        '--random_start_frame', 
        help="Use random start frame when processing videos. Good for long videos where frames have different scenes and meanings.", 
        action='store_true', 
        default=False
    )
    parser.add_argument(
        '--clip_frame_data', 
        help="Save the frames as video clips to HDD/SDD. Videos clips are saved in the same folder as your json directory.", 
        action='store_true', 
        default=False
    )
    parser.add_argument('--max_frames', help="Maximum frames for clips when --clip_frame_data is enabled.", type=int, default=60)
    parser.add_argument('--beam_amount', help="Amount for BLIP beam search.", type=int, default=7)
    parser.add_argument('--prompt_amount', help="The amount of prompts per video that is processed.", type=int, default=25)
    parser.add_argument('--min_prompt_length', help="Minimum words required in prompt.", type=int, default=15)
    parser.add_argument('--max_prompt_length', help="Maximum words required in prompt.", type=int, default=30)
    parser.add_argument('--save_dir', help="The directory to save the config to.", type=str, default=f"{os.getcwd()}/train_data")

    args = parser.parse_args()

    
    processor = PreProcessVideos(**vars(args))
    processor.process_videos()
