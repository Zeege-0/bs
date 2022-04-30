import sys
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2
import torch
import sklearn
from sklearn import metrics
from timeit import default_timer as timer


def jacard(y_pred, y_true, reduce=True):
    assert (y_pred.shape == y_true.shape) and (y_pred.ndim >= 3)
    N = y_pred.shape[0]
    y_true_f = (y_true > 0.3).view(N, -1).type_as(y_true)
    y_pred_f = (y_pred > 0.2).view(N, -1).type_as(y_true_f)
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f + y_pred_f - y_true_f * y_pred_f, dim=1) + 1e-10
    ret = intersection / union
    return ret.mean() if reduce else ret


def dice_coef(y_pred, y_true, reduce=True):
    assert (y_pred.shape == y_true.shape) and (y_pred.ndim >= 3)
    N = y_pred.shape[0]
    y_true_f = y_true.view(N, -1)
    y_pred_f = y_pred.view(N, -1)
    smooth = 1.0
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    ret = (2. * intersection + smooth) / (torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1) + smooth)
    return ret.mean() if reduce else ret


def get_seg_metrics(pred, label):
    assert (pred.shape == label.shape) and (pred.ndim >= 3)
    with torch.no_grad():
        
        ret = dict()
        pred = pred.detach().cpu()
        label = label.detach().cpu()
        linpred = pred.flatten()
        linlab = (label.flatten() > 0.3).type_as(linpred)

        ret['ap'] = metrics.average_precision_score(linlab, linpred.flatten())
        ret['auroc'] = metrics.roc_auc_score(linlab, linpred.flatten())
        ret['dice'] = dice_coef(pred, label, True)
        ret['jacard'] = jacard(pred, label, True)

        return ret


def get_metrics(pred, label, topk=(1,), train=False):
    with torch.no_grad():
        ret = dict()

        if isinstance(label, list):
            label = torch.stack(label).t()

        maxk = max(topk)
        batch_size = label.size(0)

        pred = pred.detach().cpu()
        label = label.detach().cpu()

        toppred = pred.topk(maxk, 1, True, True)[1].detach().cpu().t()
        toplab = label.topk(maxk, 1, True, True)[1].detach().cpu().t()
        correct = toppred.eq(toplab)
        linpred = toppred.view(-1)
        linlab = toplab.view(-1)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.div_(batch_size).item())
        ret['topk'] = res
        if linlab.max() != linlab.min():
            try:
                ret['ap'] = metrics.average_precision_score(label, pred)
                ret['auroc'] = metrics.roc_auc_score(label, pred)
            except ValueError:
                ret['ap'] = ret['topk'][0]
                ret['auroc'] = ret['topk'][0]
            if not train:
                ret['confusion'] = metrics.confusion_matrix(linlab, linpred)
                ret['report'] = metrics.classification_report(linlab, linpred, output_dict=True, zero_division=0)
                ret['f1'] = ret['report']['weighted avg']['f1-score']
        else:
            ret['ap'] = ret['topk'][0]
            ret['auroc'] = ret['topk'][0]
        return ret


def evaluate_metrics(res, results_path, run_name):

    img_names = res['sample_names']
    predictions = res['decs']
    labels = res['clazs']

    metrics = get_metrics(labels, predictions)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')


@np.deprecate(message="This function is deprecated. Use the function 'get_metrics' instead.")
def zzzget_metrics_deprecated(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics



def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False):
    plt.figure()
    plt.clf()
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation = np.transpose(segmentation)
        label = np.transpose(label)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth')
    plt.imshow(label, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('Output')
    else:
        plt.title(f"Output: {decision:.5f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output scaled')
    if blur:
        normed = segmentation / segmentation.max()
        blured = cv2.blur(normed, (32, 32))
        plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''

    try:
        plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        os.mkdir(save_dir)
        plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
        plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class FakeTqdm(object):
    def __init__(self, total=None, *vags, **kwargs):
        self.total = total
        self.n = 0
        self.st = timer()

    def set_postfix(self, *vargs, **kwargs):
        pass

    def update(self, n):
        self.n += n
        sys.stdout.write(f'\r{self.n}/{self.total} {timer() - self.st:.2f}s ')

    def close(self):
        print("|| ", end="")
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return f'{self.n}/{self.total}'


if __name__ == '__main__':
    bar = FakeTqdm(total=10)
    for i in range(10):
        bar.update(1)
        time.sleep(0.3)
    bar.close()
