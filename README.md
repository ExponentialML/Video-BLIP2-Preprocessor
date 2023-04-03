# Video-BLIP2-Preprocessor
A simple script that reads a directory of videos, grabs a random frame, and automatically discovers a prompt for it.
This makes unconditional/conditional video training easier to manage without manually prompting the differences in scenes.

## Installation

```bash
git clone https://github.com/ExponentialML/Video-BLIP2-Preprocessor.git
cd Video-BLIP2-Preprocessor
pip install -r requirements.txt
```

## Running

```bash
python preprocess.py --video_directory <your path of videos (vid1.mp4, vd2.mp4, etc.)> --config_name "My Videos" --config_save_name "my_videos"
```

#### If you wish to save the videos as individual clips, you can pass the `--clip_frame_data` argument like so:

```bash
python preprocess.py --clip_frame_data --video_directory 'videos' --config_name "My Videos" --config_save_name "my_videos"
```

## Results
After running, you should get a JSON like this. You can then parse it any script that supports reading JSON files.
Here is psuedo code of what your config may look like.

```js
{
    "name": "My Videos",
    "data": [
        {
            "video_path": "./videos/video.mp4",
            "num_frames": 1000,
            "data": [
                {
                    "frame_index": 134,
                    "prompt": "a person is riding a bike on a busy street.",
                    
                    // When the argument --clip_frame_data is passed.
                    // This is applied to all items in 'data', but shown once here as an example.
                    "clip_path": "./my_videos/a person is riding a bike on a busy street_134.mp4" 
                },
                {
                    "frame_index": 745,
                    "prompt": "a person is wearing a blue shirt and riding a bike on grass."
                },
                ...
            ]
        },
        ...
    ]
}
```

## Default Arguments
```py
--config_name, help="The name of the configuration.", default='My Config'

--config_save_name, help="The name of the config file that's saved.", default='my_config'

--video_directory, help="The directory where your videos are located.", default='./videos'

--clip_frame_data, help="Save the frames as video clips to HDD/SDD. Videos clips are saved in the same folder as your json directory.", default=False

--max_frames, help="Maximum frames for clips when --clip_frame_data is enabled.", default=60

--random_start_frame, 
help="Use random start frame when processing videos. Good for long videos where frames have different scenes and meanings.", 
action='store_true', 
default=True

--beam_amount, help="Amount for BLIP beam search.", default=7

--prompt_amount, help="The amount of prompts per video that is processed.", default=25

--min_prompt_length, help="Minimum words required in prompt.", default=15

--max_prompt_length, help="Maximum words required in prompt.", default=30

--save_dir, help="The directory to save the config to.", default=f"{os.getcwd()}/train_data"

```
