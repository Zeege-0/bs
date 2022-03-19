from data.input_ksdd2 import KSDD2Dataset
from config import Config

if __name__ == '__main__':
    config = Config()
    config.DATASET_PATH = '/mnt/sdb1/home/zeege/remote/bs/data/ksdd2/'
    config.INPUT_WIDTH = 229
    config.INPUT_HEIGHT = 645
    config.INPUT_CHANNELS = 3
    config.FREQUENCY_SAMPLING = True
    config.RESIZE_INPUT = False
    config.NUM_SEGMENTED = 53
    config.WEIGHTED_SEG_LOSS_MAX = 2
    config.WEIGHTED_SEG_LOSS_P = 3
    dataset = KSDD2Dataset('TRAIN', config)
    for i in range(20):
        print(dataset[i])

