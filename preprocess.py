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
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        save_dir,
        max_workers
    ):

        # Paramaters for parsing videos
        self.prompt_amount = prompt_amount
        self.video_directory = video_directory
        self.random_start_frame = random_start_frame
        self.clip_frame_data = clip_frame_data
        self.max_frames = max_frames
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.max_workers = max_workers

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
            "data": {}
        }

    # Video dict for individual videos.
    # {base_config: data -> [{video_path, num_frames, data}]}
    @staticmethod
    def build_video_config(video_path: str, num_frames: int):
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
    @staticmethod
    def video_processor(
        video_reader: VideoReader,
        num_frames: int,
        random_start_frame=True,
        frame_num=0
    ):

        frame_number = (
            random.randrange(0, int(num_frames)
                             ) if random_start_frame else frame_num
        )
        frame = video_reader[frame_number].permute(2, 0, 1)
        image = transforms.ToPILImage()(frame).convert("RGB")
        return frame_number, image

    @staticmethod
    def get_frame_range(derterministic, prompt_amount, random_start_frame):
        return range(prompt_amount) if random_start_frame else derterministic

    @staticmethod
    def process_blip(image: Image, processor, device, blip_model, beam_amount, min_length, max_length):
        inputs = processor(images=image, return_tensors="pt").to(
            device, torch.float16)
        generated_ids = blip_model.generate(
            **inputs,
            num_beams=beam_amount,
            min_length=min_length,
            max_length=max_length
        )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True)[0].strip()

        return generated_text

    @staticmethod
    def get_out_paths(prompt, frame_number, save_dir, config_save_name):
        out_name = f"{prompt}_{str(frame_number)}"
        save_path = f"{save_dir}/{config_save_name}"
        save_filepath = f"{save_path}/{out_name}.mp4"

        return out_name, save_path, save_filepath

    def save_train_config(self, config: dict):
        os.makedirs(self.save_dir, exist_ok=True)

        save_json = json.dumps(config, indent=4)
        save_dir = f"{self.save_dir}/{self.config_save_name}"

        with open(f"{save_dir}.json", 'w') as f:
            f.write(save_json)

    @staticmethod
    def save_video(save_path, save_filepath, frames):
        os.makedirs(save_path, exist_ok=True)
        torchvision.io.write_video(save_filepath, frames, fps=30)

    # Main loop for processing all videos.
    def process_videos(self):
        self.load_blip()
        config = self.build_base_config()

        if not os.path.exists(self.video_directory):
            raise ValueError(f"{self.video_directory} does not exist.")

        videos = []
        for _, _, files in tqdm(
            os.walk(self.video_directory),
            desc=f"Processing videos in {self.video_directory}"
        ):
            for video in files:
                if video.endswith(self.vid_types):
                    video_path = f"{self.video_directory}/{video}"
                    videos.append(video_path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for video_path in videos:
                future = executor.submit(PreProcessVideos.process_video, video_path, self.clip_frame_data, self.max_frames, self.prompt_amount,
                                         config, self.random_start_frame, self.processor, self.device, self.blip_model, self.beam_amount, self.min_length, self.max_length, self.save_dir, self.config_save_name)
                futures.append(future)

            for future in as_completed(futures):
                video_config = future.result()
                if video_config is not None:
                    config['data'][os.path.basename(
                        video_config['video_path'])] = video_config
        print(f"Done. Saving train config to {self.save_dir}.")
        self.save_train_config(config)

    @staticmethod
    def process_video(video_path, clip_frame_data, max_frames, prompt_amount, config, random_start_frame, processor, device, blip_model, beam_amount, min_length, max_length, save_dir, config_save_name):
        video_reader = None
        derterministic_range = None
        video_len = 0
        try:
            video_reader = VideoReader(video_path, ctx=cpu(0))
            video_len = len(video_reader)
            frame_step = abs(video_len // prompt_amount)
            derterministic_range = range(1, abs(video_len - 1), frame_step)
        except:
            print(
                f"Error loading {video_path}. Video may be unsupported or corrupt.")
            return None

        # Another try catch block because decord isn't perfect.
        try:
            num_frames = int(len(video_reader))
            video_config = PreProcessVideos.build_video_config(
                video_path, num_frames)

            # Secondary loop that process a specified amount of prompts, selects a random frame, then appends it.
            for i in tqdm(
                PreProcessVideos.get_frame_range(
                    derterministic_range, prompt_amount, random_start_frame),
                desc=f"Processing {os.path.basename(video_path)}"
            ):
                frame_number, image = PreProcessVideos.video_processor(
                    video_reader,
                    num_frames,
                    random_start_frame,
                    frame_num=i
                )

                prompt = PreProcessVideos.process_blip(
                    image, processor, device, blip_model, beam_amount, min_length, max_length)
                video_data = PreProcessVideos.build_video_data(
                    frame_number, prompt)

                if clip_frame_data:

                    # Minimum value, frame number, max value (length of entire video)
                    max_range = abs(len(video_reader) - 1)
                    frame_number = i
                    frame_number = sorted((1, frame_number, max_range))[1]

                    frame_range = range(frame_number, max_range)
                    frame_range_nums = list(frame_range)

                    frames = video_reader.get_batch(
                        frame_range_nums[:max_frames])

                    out_name, save_path, save_filepath = PreProcessVideos.get_out_paths(
                        prompt, frame_number, save_dir, config_save_name)

                    PreProcessVideos.save_video(
                        save_path, save_filepath, frames)

                    video_data['clip_path'] = save_filepath
                    video_config["data"].append(video_data)

                else:
                    video_config["data"].append(video_data)

            config['data'].append(video_config)

        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name', help="The name of the configuration.", type=str, default='My Config')
    parser.add_argument(
        '--config_save_name', help="The name of the config file that's saved.", type=str, default='my_config')
    parser.add_argument(
        '--video_directory', help="The directory where your videos are located.", type=str, default='./videos')
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
    parser.add_argument(
        '--max_frames', help="Maximum frames for clips when --clip_frame_data is enabled.", type=int, default=60)
    parser.add_argument(
        '--beam_amount', help="Amount for BLIP beam search.", type=int, default=7)
    parser.add_argument(
        '--prompt_amount', help="The amount of prompts per video that is processed.", type=int, default=25)
    parser.add_argument('--min_prompt_length',
                        help="Minimum words required in prompt.", type=int, default=15)
    parser.add_argument('--max_prompt_length',
                        help="Maximum words required in prompt.", type=int, default=30)
    parser.add_argument('--save_dir', help="The directory to save the config to.",
                        type=str, default=f"{os.getcwd()}/train_data")
    parser.add_argument(
        '--max_workers', help="Number of threads that execute this job. It defaults to 1 if not specified.", type=int, default=1)

    args = parser.parse_args()

    processor = PreProcessVideos(**vars(args))
    processor.process_videos()
