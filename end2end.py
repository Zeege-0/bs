from multiprocessing import Pool
import matplotlib
from CFPNetOrigin import create_model
from attention.segs import CFPNetRealOrigin
from attention.resori import ResnetOrigin
from attention.effori import EfficinetNetOrigin

from attention.effnet import EfficinetNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet
from liz_model import LizNet
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from tqdm import tqdm, trange
from timeit import default_timer as timer
from torch import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from sam import SAM, enable_running_stats, disable_running_stats

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 2  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False

class End2End:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)

    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):
        self._set_results_path()
        self._create_results_dirs()
        self.print_run_params()
        self._save_params()
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        device = self._get_device()
        model = self._get_model().to(device)
        # mypath = "/mnt/sdb1/home/zeege/remote/bs/doutput/STEEL/zzztransfer/models/best_47_ap0.965_f0.899.pth"
        # model = torch.load(mypath).to(device)
        # print("loaded model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        loss_seg, loss_dec = self._get_loss(True), self._get_loss(False)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        self._log("Starting training")
        train_results = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, scheduler, validation_loader, tensorboard_writer)
        self._save_train_results(train_results)
        self._save_model(model)

        self.test_model(model, device, self.cfg.SAVE_IMAGES, False, False)


    def test_model(self, model, device, save_images, plot_seg, reload_final):
        self.reload_model(model, reload_final)
        test_loader = get_dataset("TEST", self.cfg)
        self.eval_model(device, model, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg)

    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, scheduler, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index):
        _images, _claz, _seg_masks, _seg_loss_masks, _is_segmented, _ = data
        
        # Start loss function
        def loss_function(is_first_step):
            with autocast('cuda', enabled=self.cfg.USE_MIX):

                images = _images.to(device, non_blocking=True)
                seg_masks = _seg_masks.to(device, non_blocking=True)
                seg_loss_masks = _seg_loss_masks.to(device, non_blocking=True)
                claz = torch.stack(_claz).T.to(device, non_blocking=True)
                is_segmented = _is_segmented.to(device, non_blocking=True)
                
                decision, output_seg_mask = model(images)

                # fake label for non segmented images
                non_segmented_mask = ((is_segmented == False) & (claz[:, 0] == 0))
                # seg_masks[non_segmented_mask] = 1
                # output_seg_mask[non_segmented_mask] = torch.Tensor([np.finfo(np.float32).max])
                seg_loss_masks[non_segmented_mask] = 0

                if self.cfg.WEIGHTED_SEG_LOSS:
                    loss_seg = torch.mean(criterion_seg(output_seg_mask, seg_masks) * seg_loss_masks)
                else:
                    loss_seg = criterion_seg(output_seg_mask, seg_masks)

                loss_dec = criterion_dec(decision, claz.type_as(decision))
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec

                if is_first_step:
                    mts = utils.get_metrics(decision, claz, train=True)
                    _total_correct = int(mts['topk'][0] * self.cfg.BATCH_SIZE)
                    _ap = mts['ap']
                    _auroc = mts['auroc']
                    _total_loss_seg = loss_seg.item()
                    _total_loss_dec = loss_dec.item()
                    _total_loss = loss.item()
                    return loss, _total_loss, _total_correct, _total_loss_seg, _total_loss_dec, _ap, _auroc
            
            return loss
        # End loss function

        if not self.cfg.USE_SAM:
            optimizer.zero_grad()
        else:
            enable_running_stats(model)

        # First pass
        loss, tl, tc, tls, tld, tap, troc = loss_function(True)
        total_correct = tc
        total_auroc = troc
        total_ap = tap
        total_loss = tl
        total_loss_seg = tls
        total_loss_dec = tld
        loss.backward()

        # Backward and optimize
        if not self.cfg.USE_SAM:
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            loss_function(False).backward()
            optimizer.second_step(zero_grad=True)

        return total_loss_seg, total_loss_dec, total_loss, total_correct, total_ap, total_auroc

    def _train_model(self, device, model: nn.Module, train_loader, criterion_seg, criterion_dec, optimizer, scheduler, validation_set, tensorboard_writer):
        try:
            losses = []
            validation_data = []
            max_validation = -1
            validation_step = self.cfg.VALIDATION_N_EPOCHS

            num_epochs = self.cfg.EPOCHS
            samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE

            self.set_dec_gradient_multiplier(model, 0.0)

            for epoch in range(num_epochs):

                model.train()

                weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
                dec_gradient_multiplier = self.get_dec_gradient_multiplier()
                self.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

                epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
                epoch_correct = 0
                epoch_ap = 0
                epoch_auroc = 0

                pbar = tqdm(total=len(train_loader), ncols=140, bar_format="{desc} {bar}{r_bar}")
                time_acc = 0
                start = timer()
                for iter_index, (data) in enumerate(train_loader):
                    pbar.set_description(f"{epoch + 1}/{num_epochs}")
                    start_1 = timer()
                    curr_loss_seg, curr_loss_dec, curr_loss, correct, ap, auroc = self.training_iteration(data, device, model,
                                                                                            criterion_seg,
                                                                                            criterion_dec,
                                                                                            optimizer, 
                                                                                            scheduler,
                                                                                            weight_loss_seg,
                                                                                            weight_loss_dec,
                                                                                            tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                    end_1 = timer()
                    time_acc = time_acc + (end_1 - start_1)

                    epoch_loss_seg += curr_loss_seg
                    epoch_loss_dec += curr_loss_dec
                    epoch_loss += curr_loss
                    epoch_correct += int(correct)
                    epoch_ap += ap
                    epoch_auroc += auroc
                    pbar.update(1)
                    pbar.set_postfix({
                        # 'time': f'{time_acc:.2f}s',
                        'top1': f"{epoch_correct}/{(iter_index + 1) * self.cfg.BATCH_SIZE}",
                        'ap': f"{(epoch_ap / (iter_index + 1)):.2f}",
                        'auroc': f"{(epoch_auroc / (iter_index + 1)):.2f}",
                        'seg': f"{(epoch_loss_seg / (iter_index + 1) ):.5f}",
                        'dec': f"{(epoch_loss_dec / (iter_index + 1)):.5f}",
                        'loss': f"{(epoch_loss / (iter_index + 1)):.5f}",
                    })

                end = timer()
                pbar.close()

                epoch_loss_seg = epoch_loss_seg / len(train_loader)
                epoch_loss_dec = epoch_loss_dec / len(train_loader)
                epoch_loss = epoch_loss / len(train_loader)
                scheduler.step(epoch_loss)
                losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))

                # self._log(f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                    tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                    tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                    tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

                if (epoch > 0.3 * num_epochs) and (epoch % 5 == 0 or epoch == num_epochs - 1):
                    self._save_model(model, f"ep_{epoch:02}.pth")

                if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):

                    mts = self.eval_model(device, model, validation_set, None, False, True, False)
                    report = mts['report']
                    confu = mts['confusion']
                    validation_ap = mts['ap']
                    validation_accuracy = mts['topk'][0]
                    validation_f1 = mts['f1']
                    validation_data.append((
                        validation_ap, 
                        epoch, 
                        mts['f1'], 
                        confu[0][1], # FP
                        confu[1][0], # FN
                        confu[1][1], # TP
                        confu[0][0] # TN
                    ))

                    if validation_ap > max_validation:
                        max_validation = validation_ap
                        name = f"best_{epoch:02}_ap{validation_ap:.3f}_f{validation_f1:.3f}.pth"
                        self._log(f"Saving model {name}")
                        self._save_model(model, name)

                    model.train()
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)
        except KeyboardInterrupt:
            self._log("Training interrupted")
            self._save_model(model, "interrupt.pth")
            return losses, validation_data
        return losses, validation_data

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg):
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
            
            if is_validation:
                pbar = utils.FakeTqdm(total=len(eval_loader))
            else:
                pbar = tqdm(total=len(eval_loader), ncols=80)
            time_acc = 1e-10
            for iii, (data_point) in enumerate(eval_loader):
                image, claz, seg_mask, seg_loss_mask, _, sample_name = data_point
                image = image.to(device, non_blocking=True)
                claz = torch.stack(claz).T

                start = timer()
                prediction, pred_seg = model(image)
                end = timer()
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
        self._log(f"|| AUROC: {mts['auroc']:.4f}, AP: {mts['ap']:.4f}, ACC: {mts['topk'][0]:.4f}, F1: {mts['f1']:.4f} || FP: {FP:d}, FN: {FN:d}"
                  f" || AUROC: {segmts['auroc']:.4f}, AP: {segmts['ap']:.4f}, DICE: {segmts['dice']:.4f}, JCCARD: {segmts['jacard']:.4f}")
        
        if not is_validation:
            torch.save(mts, f"{save_folder}/metrics.pth")
            preds = res['decs'].max(dim=1)[1].type(torch.int)
            gts = res['clazs'].max(dim=1)[1].type(torch.int)
            df = pd.DataFrame(data={
                'decision': preds,
                'ground_truth': gts,
                'img_name': res['sample_names']
            })
            df = pd.concat([df, pd.DataFrame(res['decs'])], axis=1)
            df.to_csv(os.path.join(self.run_path, 'results.csv'), index=False)

            if save_images:
                dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT
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
                    if False: # self.cfg.WEIGHTED_SEG_LOSS:
                        seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                        utils.plot_sample(sample_name, image, pred_seg, seg_loss_mask, save_folder, decision=prediction, plot_seg=plot_seg)
                    else:
                        utils.plot_sample(sample_name, image, pred_seg, seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)

        return mts

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)
        alpha = 0.3

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = (1 - (epoch / total_epochs)) * (1 - 2 * alpha) + alpha
            dec_loss_weight = ((epoch / total_epochs) * (1 - 2 * alpha) + alpha) * self.cfg.DELTA_CLS_LOSS
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS

        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model=None, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            name = sorted(glob(os.path.join(self.model_path, "best_*.pth")))[-1]
            path = os.path.join(self.model_path, name)
            model.load_state_dict(torch.load(path).state_dict())
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path).state_dict())
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data = results
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid()
        plt.xlabel("Epochs")
        v, ve, f1, FP, FN, TP, TN = map(list, zip(*validation_data))
        plt.twinx()
        plt.plot(ve, v, label="Val AP", color="Green")
        plt.plot(ve, f1, label="Val F1", color="Blue")
        plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val.pdf"))

        df_loss = pd.DataFrame(data={"epoch": le, "loss_seg": ls, "loss_dec": ld, "loss": l})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)
        df_val = pd.DataFrame(data={"AP": v, "F1": f1, "FP": FP, "FN": FN, "TP": TP, "TN": TN})
        df_val.to_csv(os.path.join(self.run_path, "validation.csv"), index=False)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        # self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model, output_name)

    def _get_optimizer(self, model):
        if self.cfg.USE_SAM:
            base_optimizer = torch.optim.SGD
            return SAM(model.parameters(), base_optimizer, lr=self.cfg.LEARNING_RATE, momentum=0.9)
        else:
            return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, threshold=0.0001, verbose=True)

    def _get_loss(self, is_seg):
        reduce = "none" if is_seg and self.cfg.WEIGHTED_SEG_LOSS else "mean"
        return nn.BCEWithLogitsLoss(reduction=reduce)

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        if self.cfg.MODEL == 'MY':
            seg_net = LizNet(self.cfg.USE_MED, self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS, classes=2)
        elif self.cfg.MODEL == 'EFF':
            seg_net = EfficinetNet(self.cfg.INPUT_CHANNELS, num_classes=2)
        elif self.cfg.MODEL == 'SD':
            seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS, classes=2)
        elif self.cfg.MODEL == 'EFF_ORI':
            seg_net = EfficinetNetOrigin(self.cfg.INPUT_CHANNELS, num_classes=2)
        elif self.cfg.MODEL == 'RES_ORI':
            seg_net = ResnetOrigin(pretrained=False, num_classes=2)
        elif self.cfg.MODEL == 'CFP_ORI':
            seg_net = CFPNetRealOrigin(self.cfg.INPUT_CHANNELS)
        else:
            raise NotImplementedError(f"Model {self.cfg.MODEL} not implemented")
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
