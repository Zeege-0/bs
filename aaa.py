from tqdm import tqdm, trange
from models import SegDecNet
from liz_model import LizNet
from data.input_ksdd2 import KSDD2Dataset
from data.dataset_catalog import get_dataset
from config import Config
import torch
import torch.nn as nn
from timeit import default_timer as timer
import utils
import numpy as np
import thop
import pandas as pd
import os
import cv2
import torch.nn.functional as F


if __name__ == '__main__':
    device = 'cuda:0'
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/dzfinal/STEEL/zzzmy_3000_3000/models/ep_90.pth').to(device)
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/STEEL/simam_3000_3000/models/ep_90.pth').to(device)
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/STEEL/my_1500_1500/models/best_state_dict.pth').to(device)
    # model = SegDecNet(device, 1600, 256, 1, 1, False).to(device)
    model = torch.load('/mnt/sdb1/home/zeege/remote/bs/dzfinal/KSDD2/resori/models/ep_25.pth').to(device)
    # model = LizNet(False, device, 1600, 256, 1, 1).to(device)
    # model.set_gradient_multipliers(0)
    # torch.save(model, "/mnt/sdb1/home/zeege/remote/bs/model.pth")
    model.eval()
    with torch.no_grad():
        data = torch.randn(64, 1, 1600, 256).to(device)
        model(data)
        flops, params = thop.profile(model, inputs=(torch.randn(1, 1, 1600, 256).to(device),), verbose=False)
        flops, params = thop.clever_format([flops, params], "%.6f")
        print(flops, params, end=' ')
        acc = 0
        model(data)
        for i in range(20):
            data = torch.randn(64, 1, 1600, 256).to(device)
            st = timer()
            model(data)
            ed = timer()
            acc += ed - st
            print(ed - st, end=', ')
        print(64 * 20 / acc)


if __name__ == '__main__':
    config = Config()
    config.DATASET_PATH = '/mnt/sdb1/home/zeege/remote/bs/data/ksdd2/'
    config.FREQUENCY_SAMPLING = True
    config.RESIZE_INPUT = False
    config.NUM_SEGMENTED = 246
    config.TRAIN_NUM = 246
    config.WEIGHTED_SEG_LOSS_MAX = 1
    config.WEIGHTED_SEG_LOSS_P = 0
    config.DATASET = 'KSDD2'
    config.BATCH_SIZE = 64
    config.DILATE = 0
    config.init_extra()
    eval_loader = get_dataset('TEST', config)
    device = 'cuda:1'
    save_folder = "/mnt/sdb1/home/zeege/remote/bs/zfora"
    # model = LizNet(False, 'cuda:0', config.INPUT_WIDTH, config.INPUT_HEIGHT, config.INPUT_CHANNELS).to(device)
    # model.load_state_dict(torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/KSDD2/att_246_b816068ec/models/ep_75.pth').state_dict())
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/dzfinal/STEEL/zzzmy_3000_3000/models/ep_90.pth').to(device)
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/dzfinal/STEEL/gct/models/best_44_ap0.957_f0.896.pth').to(device)
    # model = torch.load('/mnt/sdb1/home/zeege/remote/bs/dzfinal/STEEL/upsample_inf/models/best_65_ap0.946_f0.871.pth').to(device)
    model = torch.load('/mnt/sdb1/home/zeege/remote/bs/doutput/KSDD2/playground/models/best_48_ap0.940_f0.954.pth').to(device)
    model.device = device
    model.set_gradient_multipliers(0)
    predictions = []
    ground_truths = []
    res = []
    model.eval()
    for i in range(1):
        pbar = tqdm(eval_loader, ncols=80)
        time_acc = 0
        with torch.no_grad():
            model.eval()
            res = {
                'sample_names': [],
                'decs': [],
                'clazs': [],
                'pred_segs': [],
                'true_segs': [],
                'images': []
            }
            time_acc = 1e-10
            for iii, (data_point) in enumerate(eval_loader):
                image, claz, seg_mask, seg_loss_mask, _, sample_name = data_point
                image = image.to(device, non_blocking=True)
                claz = torch.stack(claz).T

                start = timer()
                prediction, pred_seg = model(image)
                end = timer()

                # pred_seg, volume = model.volume(image)
                # pred_seg = model.seg_mask(pred_seg)
                # pred_seg = F.interpolate(pred_seg, seg_mask.shape[2:], mode='bilinear', align_corners=False)

                if iii > 1:
                    time_acc = time_acc + (end - start)

                pred_seg = nn.Sigmoid()(pred_seg)
                prediction = nn.Sigmoid()(prediction)

                prediction = prediction.detach().cpu()
                image = image.detach().cpu().numpy()
                pred_seg = pred_seg.detach().cpu().numpy()
                seg_mask = seg_mask.detach().cpu().numpy()

                res["sample_names"].extend(sample_name)
                res["decs"].extend(prediction)
                res["clazs"].extend(claz)
                res["pred_segs"].extend(pred_seg)
                res["true_segs"].extend(seg_mask)
                res["images"].extend(image)
                pbar.update(1)
                pbar.set_postfix({
                    "FPS": iii / time_acc * image.shape[0]
                })
            pbar.close()

        res["decs"] = torch.stack(res["decs"])
        res["clazs"] = torch.stack(res["clazs"])
        res["pred_segs"] = torch.Tensor(np.array(res['pred_segs']))
        res["true_segs"] = torch.Tensor(np.array(res['true_segs']))
        res["images"] = torch.Tensor(np.array(res['images']))

        mts = utils.get_metrics(res['decs'], res['clazs'])
        segmts = utils.get_seg_metrics(res['pred_segs'], res['true_segs'])
        mts["seg"] = segmts
        confu = mts['confusion']
        FP = confu[0][1]
        FN = confu[1][0]
        print(f"VALIDATION || AUROC: {mts['auroc']:.4f}, AP: {mts['ap']:.4f}, ACC: {mts['topk'][0]:.4f}, F1: {mts['f1']:.4f} || FP: {FP:d}, FN: {FN:d}"
                  f" || AUROC: {segmts['auroc']:.4f}, AP: {segmts['ap']:.4f}, DICE: {segmts['dice']:.4f}, JCCARD: {segmts['jacard']:.4f}")
        
        pbar.update(1)
        if iii > 1:
            pbar.set_postfix({"fps": iii * config.BATCH_SIZE / time_acc})
        pbar.close()

    torch.save(mts, f"{save_folder}/metrics.pth")
    preds = res['decs'].max(dim=1)[1].type(torch.int)
    gts = res['clazs'].max(dim=1)[1].type(torch.int)
    df = pd.DataFrame(data={
        'decision': preds,
        'ground_truth': gts,
        'img_name': res['sample_names']
    })
    df = pd.concat([df, pd.DataFrame(res['decs'])], axis=1)
    df.to_csv("/mnt/sdb1/home/zeege/remote/bs/zfora/results.csv", index=False)
    # exit(1)

    dsize = config.INPUT_WIDTH, config.INPUT_HEIGHT
    for idx in trange(len(res["clazs"]), ncols=80):
        image = res['images'][idx].permute(1, 2, 0).numpy()
        pred_seg = res['pred_segs'][idx].permute(1, 2, 0).numpy()
        seg_mask = res['true_segs'][idx].permute(1, 2, 0).numpy()
        sample_name = res['sample_names'][idx]
        prediction = preds[idx].item()
        image = cv2.resize(image, dsize)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pred_seg = cv2.resize(pred_seg, dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg, dsize)
        seg_mask = cv2.resize(seg_mask, dsize)
        utils.plot_sample(sample_name, image, pred_seg, seg_mask, save_folder, decision=prediction, plot_seg=False)

