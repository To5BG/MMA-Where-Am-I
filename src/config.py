from dataclasses import dataclass


@dataclass
class Config:
    """Config"""
    root_data: str = "../data/imagefolders"
    resize_width: int = 1000


config = Config()

