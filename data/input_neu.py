import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
from tqdm import tqdm


class NEUDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(NEUDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        print(f"Loading NEU {kind} dataset...")
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

