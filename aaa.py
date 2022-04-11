from tqdm import tqdm
from liz_model import LizNet
from data.input_ksdd2 import KSDD2Dataset
from data.dataset_catalog import get_dataset
from config import Config
import torch
import torch.nn as nn
from timeit import default_timer as timer
import utils

if __name__ == '__main__':
    config = Config()
    config.DATASET_PATH = '/mnt/sdb1/home/zeege/remote/bs/data/ksdd2/'
    config.INPUT_WIDTH = 232
    config.INPUT_HEIGHT = 640
    config.INPUT_CHANNELS = 3
    config.FREQUENCY_SAMPLING = True
    config.RESIZE_INPUT = False
    config.NUM_SEGMENTED = 246
    config.WEIGHTED_SEG_LOSS_MAX = 2
    config.WEIGHTED_SEG_LOSS_P = 3
    config.DATASET = 'KSDD2'
    config.BATCH_SIZE = 64
    eval_loader = get_dataset('TRAIN', config)
    device = 'cuda:0'
    # model = LizNet(False, 'cuda:0', config.INPUT_WIDTH, config.INPUT_HEIGHT, config.INPUT_CHANNELS).to(device)
    # model.load_state_dict(torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/KSDD2/att_246_b816068ec/models/ep_75.pth').state_dict())
    model = torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/KSDD2/att_246_b816068ec/models/ep_75.pth').to(device)
    predictions = []
    ground_truths = []
    res = []
    model.eval()
    for i in range(5):
        pbar = tqdm(eval_loader, ncols=80)
        time_acc = 0
        with torch.autocast('cuda'):
            with torch.no_grad():
                for iii, (data_point) in enumerate(eval_loader):
                    image, claz, seg_mask, seg_loss_mask, _, sample_name = data_point
                    image, seg_mask = image.to(device), seg_mask.to(device)
                    is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item()

                    start = prediction = pred_seg = end = None

                    start = timer()
                    prediction, pred_seg = model(image)
                    end = timer()
                    if iii > 1:
                        time_acc = time_acc + (end - start)

                    # pred_seg = nn.Sigmoid()(pred_seg)
                    # prediction = nn.Sigmoid()(prediction)

                    # prediction = prediction.item()
                    # image = image.detach().cpu().numpy()
                    # pred_seg = pred_seg.detach().cpu().numpy()
                    # seg_mask = seg_mask.detach().cpu().numpy()

                    # predictions.append(prediction)
                    # ground_truths.append(is_pos)
                    # res.append((prediction, None, None, is_pos, sample_name[0]))
                    pbar.update(1)
                    if iii > 1:
                        pbar.set_postfix({"fps": iii * config.BATCH_SIZE / time_acc})
            pbar.close()

    # utils.evaluate_metrics(res, '/mnt/sdb1/home/zeege/remote/bs/doutput', 'att_246_b816068ec')
    print("===============================================")
    print("===============================================")
    for i in range(5):
        pbar = tqdm(eval_loader, ncols=80)
        time_acc = 0
        with torch.no_grad():
            for iii, (data_point) in enumerate(eval_loader):
                image, claz, seg_mask, seg_loss_mask, _, sample_name = data_point
                image, seg_mask = image.to(device), seg_mask.to(device)
                is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item()

                start = prediction = pred_seg = end = None

                start = timer()
                prediction, pred_seg = model(image)
                end = timer()
                if iii > 1:
                    time_acc = time_acc + (end - start)

                # pred_seg = nn.Sigmoid()(pred_seg)
                # prediction = nn.Sigmoid()(prediction)

                # prediction = prediction.item()
                # image = image.detach().cpu().numpy()
                # pred_seg = pred_seg.detach().cpu().numpy()
                # seg_mask = seg_mask.detach().cpu().numpy()

                # predictions.append(prediction)
                # ground_truths.append(is_pos)
                # res.append((prediction, None, None, is_pos, sample_name[0]))
                pbar.update(1)
                if iii > 1:
                    pbar.set_postfix({"fps": iii * config.BATCH_SIZE / time_acc})
        pbar.close()

