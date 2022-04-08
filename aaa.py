from data.dataset_catalog import get_dataset
from config import Config

if __name__ == '__main__':
    config = Config()
    config.DATASET_PATH = '/mnt/sdb1/home/zeege/remote/bs/data/steel/'
    config.INPUT_WIDTH = 1600
    config.INPUT_HEIGHT = 256
    config.INPUT_CHANNELS = 3
    config.FREQUENCY_SAMPLING = True
    config.RESIZE_INPUT = False
    config.NUM_SEGMENTED = 300
    config.TRAIN_NUM = 300
    config.WEIGHTED_SEG_LOSS_MAX = 1
    config.WEIGHTED_SEG_LOSS_P = 2
    config.BATCH_SIZE = 32
    config.DATASET = 'STEEL'
    # train = get_dataset('TRAIN', config)
    # val = get_dataset('VAL', config)
    test = get_dataset('TEST', config)
    for i in test:
        if i[1] == 1:
            print('aaa')

