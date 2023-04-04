from dataclasses import dataclass

@dataclass
class Config:
    """Config"""
    root_data: str = "../data/imagefolders"
    video_path: str = "../data/imagefolders/test_videos"
    resize_width: int = 1000
    visual_words: int = 12
    kneighbours: int = 7
    video_sample_rate: int = 10

config = Config()