import os
import numpy as np
from data.dataset import Dataset
import pickle
import pandas as pd


def read_split(train_num: int, num_segmented: int, kind: str):
    fn = f"STEEL/split_{train_num}_{num_segmented}.pyb"
    with open(f"/mnt/sdb1/home/zeege/remote/bs/KSDD2/splits/{fn}", "rb") as f:
        train_samples, test_samples, validation_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        elif kind == 'VAL':
            return validation_samples
        else:
            raise Exception('Unknown')


def read_annotations(fn):
    arr = np.array(pd.read_csv(fn), dtype=np.object)
    annotations_dict = {}
    for sample, _, rle in arr:
        img_name = sample[:-4]
        annotations_dict[img_name] = rle

    return annotations_dict


class SteelDataset(Dataset):
    def __init__(self, kind, cfg):
        super(SteelDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):

        pos_samples, neg_samples = [], []

        fn = os.path.join(self.path, "train.csv")
        annotations = read_annotations(fn)

        samples = read_split(self.cfg.TRAIN_NUM, self.cfg.NUM_SEGMENTED, self.kind)
        for sample, is_segmented in samples:
            img_name = f"{sample}.jpg"
            img_path = os.path.join(self.path, "train_images", img_name)

            if sample in annotations:
                rle = list(map(int, annotations[sample].split(" ")))
                img = self.read_img_resize(img_path, self.grayscale, self.image_size)                    
                seg_mask = self.rle_to_mask(rle, self.image_size)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                image = self.to_tensor(img)
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, img_path, rle, sample)) # Image, segmask, seglossmask
            else:
                seg_mask = np.zeros_like(img)
                seg_loss_mask = np.ones_like(seg_mask)
                image = self.to_tensor(img)
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, img_path, None, sample))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
