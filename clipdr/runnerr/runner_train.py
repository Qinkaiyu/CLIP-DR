import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from clipdr.models import MODELS
from clipdr.models.ordinalclip import CLIPDR
from clipdr.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .fds import FDS

logger = get_logger(__name__)


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()
        self.module = MODELS.build(model_cfg)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.num_ranks = self.module.num_ranks
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.FDS = FDS(feature_dim = 5,bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9)
        self.list_logits = []
        self.list_y = []
        self.training_step_outputs = []

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()


    # Running Steps
    def run_step(self, batch, batch_idx,M):

        x, y = batch


        our_logits, image_features, text_features = self.module(x)
        our_logits = our_logits.float()

        #print("our_logits",our_logits)
        #our_logits = smoothed_features
        if M == 0:
            # similarity smooth
            #current_epoch = self.current_epoch
            #self.FDS.update_last_epoch_stats(current_epoch)
            #our_logits_copy = our_logits.detach()
            #self.FDS.update_running_stats(our_logits_copy, y, current_epoch)
            #smoothed_features = self.FDS.smooth(features=our_logits, labels=y, epoch=0)
            #our_logits = smoothed_features
            #print("smoothed_features", smoothed_features)

            rank_loss = self.rank_loss(our_logits,y)
            #loss_main
            loss_kl = self.compute_kl_loss(our_logits, y)
            loss_ce = self.ce_loss_func(our_logits, y)
            print("----------------------------------rank_loss--------------------",rank_loss)
            loss = loss_ce + loss_kl + rank_loss

            metrics_exp = self.compute_per_example_metrics(our_logits, y, "exp")
            metrics_max = self.compute_per_example_metrics(our_logits, y, "max")
            print("------------------------------------------------------------------")
            print("y",y)
            print("our_logits", our_logits)
            print("loss",loss)
        else:
            losses = self.compute_losses(our_logits, y)
            loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

            metrics_exp = self.compute_per_example_metrics(our_logits, y, "exp")
            metrics_max = self.compute_per_example_metrics(our_logits, y, "max")
        return {"loss": loss, **metrics_exp, **metrics_max}


    def training_step(self, batch, batch_idx):


        outputs = self.run_step(batch, batch_idx,0)


        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx,1)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx,0)

        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)

        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")


    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        self.list_y = []
        self.list_logits = []
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    # Loss & Metrics
    def compute_losses(self, logits, y):
        losses = {}
        losses["ce_loss"] = self.ce_loss_func(logits, y)
        losses["kl_loss"] = self.compute_kl_loss(logits, y)

        return losses

    def compute_kl_loss(self, logits, y):
        y_t = F.one_hot(y, self.num_ranks).t()
        #print("y_t",y_t)
        y_t_row_ind = y_t.sum(-1) > 0
        #print("y_t_row_ind",y_t_row_ind)

        num_slots = y_t_row_ind.sum()
        #print("num_slots",num_slots)

        y_t_reduction = (y_t * 10.0).softmax(-1)
        #print("y_t_reduction",y_t_reduction)

        y_t_reduction[y_t_row_ind <= 0] = 0
        #print("y_t_reduction",y_t_reduction)

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        dtype = logits.dtype
        probs = F.softmax(logits, -1)

        if gather_type == "exp":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        count_y_0 = (y == 0).sum().item()
        count_y_1 = (y == 1).sum().item()
        count_y_2 = (y == 2).sum().item()
        count_y_3 = (y == 3).sum().item()
        count_y_4 = (y == 4).sum().item()
        print("count_y_0",count_y_0)
        print("count_y_1",count_y_1)

        print("count_y_2",count_y_2)
        print("count_y_3",count_y_3)
        print("count_y_4",count_y_4)

        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)
        auc_ovo = roc_auc_score(y.cpu().numpy(), probs.cpu().detach().numpy(), average='macro', multi_class='ovo',labels=[0, 1, 2,3,4])
        #auc_ovo = auc_ovo*len(y)
        auc_ovo = torch.tensor(auc_ovo)
        if torch.isnan(auc_ovo):
            # 处理NaN值的情况，例如赋予一个默认值或者执行其他的操作
            print("Encountered NaN in test_exp_DGDR_auc_metric. Handling NaN...")
            auc_ovo = torch.tensor(1)
        #DGDR_acc = accuracy_score(y.cpu().numpy(), predict_y.cpu().numpy())

        f1 = f1_score(y.cpu().numpy(), torch.round(predict_y).cpu().detach().numpy(), average='macro')
        #f1 = f1*len(y)
        f1 = torch.tensor(f1)
        y_copy = y.cpu().numpy()
        probs_copy = probs.detach().cpu().numpy()
        fpr0, tpr0, _ = roc_curve(y_copy == 0, probs_copy[:, 0])
        roc_auc0 = auc(fpr0, tpr0)
        roc_curves0 = (fpr0, tpr0, roc_auc0)
        roc_auc0 = torch.tensor(roc_auc0)
        fpr1, tpr1, _ = roc_curve(y_copy == 1, probs_copy[:, 1])
        roc_auc1 = auc(fpr1, tpr1)
        roc_curves1 = (fpr1, tpr1, roc_auc1)
        roc_auc1 = torch.tensor(roc_auc1)

        fpr2, tpr2, _ = roc_curve(y_copy == 2, probs_copy[:, 2])
        roc_auc2 = auc(fpr2, tpr2)
        roc_curves2 = (fpr2, tpr2, roc_auc2)
        roc_auc2 = torch.tensor(roc_auc2)
        fpr3, tpr3, _ = roc_curve(y_copy == 3, probs_copy[:, 3])
        roc_auc3 = auc(fpr3, tpr3)
        roc_curves3 = (fpr3, tpr3, roc_auc3)
        roc_auc3 = torch.tensor(roc_auc3)

        fpr4, tpr4, _ = roc_curve(y_copy == 4, probs_copy[:, 4])
        roc_auc4 = auc(fpr4, tpr4)
        roc_curves4 = (fpr4, tpr4, roc_auc4)
        roc_auc4 = torch.tensor(roc_auc4)


        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, f"{gather_type}_DGDR_auc_metric": auc_ovo, f"{gather_type}_DGDR_f1_metric": f1,f"{gather_type}_roc_curves0_metric":roc_auc0,f"{gather_type}_roc_curves1_metric":roc_auc1,f"{gather_type}_roc_curves2_metric":roc_auc2,f"{gather_type}_roc_curves3_metric":roc_auc3,f"{gather_type}_roc_curves4_metric":roc_auc4}

    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)
    def rank_loss(self, our_logits,y):
        indexA = torch.nonzero(y == 0, as_tuple=True)[0]
        indexB = torch.nonzero(y == 1, as_tuple=True)[0]
        indexC = torch.nonzero(y == 2, as_tuple=True)[0]
        indexD = torch.nonzero(y == 3, as_tuple=True)[0]
        indexF = torch.nonzero(y == 4, as_tuple=True)[0]
        #training label
        images_similarity = torch.zeros(len(y), 5)
        texts_similarity = torch.zeros(len(y), 5)
        images_similarity1 = torch.zeros(len(y), 5)
        texts_similarity1 = torch.zeros(len(y), 5)
        images_similarity2 = torch.zeros(len(y), 5)
        texts_similarity2 = torch.zeros(len(y), 5)
        texts_similarity3 = torch.zeros(len(y), 5)
        images_similarity3 = torch.zeros(len(y), 5)
        #training target
        logits_similarity_image1 = torch.zeros(len(y), 5)
        logits_similarity_text1 = torch.zeros(len(y), 5)
        logits_similarity_image2 = torch.zeros(len(y), 5)
        logits_similarity_text2 = torch.zeros(len(y), 5)
        logits_similarity_image3 = torch.zeros(len(y), 5)
        logits_similarity_text3 = torch.zeros(len(y), 5)
        logits_similarity_image4 = torch.zeros(len(y), 5)
        logits_similarity_text4 = torch.zeros(len(y), 5)

        for i in range(len(indexA)):
            index = indexA[i]
            images_similarity[index][0] = 1
            images_similarity1[index][0] = 1
            images_similarity2[index][1] = 1
            images_similarity3[index][2] = 1

            texts_similarity[index][0] = 1
            texts_similarity1[index][0] = 1

            AA_matrix = our_logits[index][0]
            AB_matrix = our_logits[index][1]
            AC_matrix = our_logits[index][2]
            AD_matrix = our_logits[index][3]
            AF_matrix = our_logits[index][4]
            logits_similarity_image1[index][0] = AA_matrix
            logits_similarity_image1[index][1] = AB_matrix
            logits_similarity_text1[index][0] = AA_matrix
            logits_similarity_image2[index][1] = AB_matrix
            logits_similarity_image2[index][2] = AC_matrix
            logits_similarity_image3[index][2] = AC_matrix
            logits_similarity_image3[index][3] = AD_matrix

        for i in range(len(indexB)):
            index = indexB[i]
            images_similarity[index][1] = 1
            images_similarity1[index][1] = 1
            images_similarity2[index][2] = 1
            images_similarity3[index][3] = 1
            texts_similarity[index][1] = 1
            texts_similarity1[index][1] = 1
            texts_similarity2[index][0] = 1
            # image_similarity4[index][3] = 1
            BA_matrix = our_logits[index][0]
            BB_matrix = our_logits[index][1]
            BC_matrix = our_logits[index][2]
            BD_matrix = our_logits[index][3]
            BF_matrix = our_logits[index][4]
            logits_similarity_image1[index][1] = BB_matrix
            logits_similarity_image1[index][2] = BC_matrix
            logits_similarity_text1[index][1] = BB_matrix
            logits_similarity_text1[index][0] = BA_matrix
            logits_similarity_image2[index][2] = BC_matrix
            logits_similarity_image2[index][3] = BD_matrix
            logits_similarity_text2[index][0] = BA_matrix
            logits_similarity_image3[index][3] = BD_matrix
            logits_similarity_image3[index][4] = BF_matrix

        for i in range(len(indexC)):
            index = indexC[i]
            images_similarity[index][2] = 1
            images_similarity1[index][2] = 1
            images_similarity2[index][3] = 1
            images_similarity3[index][4] = 1
            texts_similarity[index][2] = 1
            texts_similarity1[index][2] = 1
            texts_similarity2[index][1] = 1
            texts_similarity3[index][0] = 1
            # image_similarity4[index][4] = 1
            # text_similarity4[index][0] = 1
            CA_matrix = our_logits[index][0]
            CB_matrix = our_logits[index][1]
            CC_matrix = our_logits[index][2]
            CD_matrix = our_logits[index][3]
            CF_matrix = our_logits[index][4]
            logits_similarity_image1[index][2] = CC_matrix
            logits_similarity_image1[index][3] = CD_matrix
            logits_similarity_text1[index][2] = CC_matrix
            logits_similarity_text1[index][1] = CB_matrix
            logits_similarity_image2[index][3] = CD_matrix
            logits_similarity_image2[index][4] = CF_matrix
            logits_similarity_text2[index][1] = CB_matrix
            logits_similarity_text2[index][0] = CA_matrix
            logits_similarity_image3[index][4] = CF_matrix
            logits_similarity_text3[index][0] = CA_matrix

        for i in range(len(indexD)):
            index = indexD[i]
            images_similarity[index][3] = 1
            images_similarity1[index][3] = 1
            images_similarity2[index][4] = 1
            texts_similarity[index][3] = 1
            texts_similarity1[index][3] = 1
            texts_similarity2[index][2] = 1
            texts_similarity3[index][1] = 1
            DA_matrix = our_logits[index][0]
            DB_matrix = our_logits[index][1]
            DC_matrix = our_logits[index][2]
            DD_matrix = our_logits[index][3]
            DF_matrix = our_logits[index][4]
            logits_similarity_image1[index][3] = DD_matrix
            logits_similarity_image1[index][4] = DF_matrix
            logits_similarity_text1[index][3] = DD_matrix
            logits_similarity_text1[index][2] = DC_matrix
            logits_similarity_image2[index][4] = DF_matrix
            logits_similarity_text2[index][2] = DC_matrix
            logits_similarity_text2[index][1] = DB_matrix
            logits_similarity_text3[index][1] = DB_matrix
            logits_similarity_text3[index][0] = DA_matrix

        for i in range(len(indexF)):
            index = indexF[i]
            images_similarity[index][4] = 1
            images_similarity1[index][4] = 1
            texts_similarity[index][4] = 1
            texts_similarity1[index][4] = 1
            texts_similarity2[index][3] = 1
            texts_similarity3[index][2] = 1
            FA_matrix = our_logits[index][0]
            FB_matrix = our_logits[index][1]
            FC_matrix = our_logits[index][2]
            FD_matrix = our_logits[index][3]
            FF_matrix = our_logits[index][4]
            logits_similarity_image1[index][4] = FF_matrix
            logits_similarity_text1[index][4] = FF_matrix
            logits_similarity_text1[index][3] = FD_matrix
            logits_similarity_text2[index][3] = FD_matrix
            logits_similarity_text2[index][2] = FC_matrix
            logits_similarity_text3[index][2] = FC_matrix
            logits_similarity_text3[index][1] = FB_matrix
        device = 'cuda:0'
        our_logits = our_logits.to(device)
        images_similarity1 = images_similarity1.to(device)
        images_similarity2 = images_similarity2.to(device)
        images_similarity3 = images_similarity3.to(device)
        texts_similarity1 = texts_similarity1.to(device)
        texts_similarity2 = texts_similarity2.to(device)
        texts_similarity3 = texts_similarity3.to(device)
        logits_similarity_text1 = logits_similarity_text1.to(device)
        logits_similarity_text2 = logits_similarity_text2.to(device)
        logits_similarity_text3 = logits_similarity_text3.to(device)
        logits_similarity_image1 = logits_similarity_image1.to(device)
        logits_similarity_image2 = logits_similarity_image2.to(device)
        logits_similarity_image3 = logits_similarity_image3.to(device)
        rank_image_loss1 = nn.CrossEntropyLoss()(logits_similarity_image1, images_similarity1)
        rank_text_loss1 = nn.CrossEntropyLoss()(logits_similarity_text1, texts_similarity1)
        rank_image_loss2 = nn.CrossEntropyLoss()(logits_similarity_image2, images_similarity2)
        rank_text_loss2 = nn.CrossEntropyLoss()(logits_similarity_text2, texts_similarity2)
        rank_image_loss3 = nn.CrossEntropyLoss()(logits_similarity_image3, images_similarity3)
        rank_text_loss3 = nn.CrossEntropyLoss()(logits_similarity_text3, texts_similarity3)
        #Rank1 is used to calculate the adjacency of AB, BC, CD, DF.
        #Rank2 is used to calculate the adjacency of AC, BD, CF.
        #Rank3 is used to calculate the adjacency of AD and BF.
        #Usually, it is sufficient to use only rank1.
        rank_loss1 = rank_image_loss1 + rank_text_loss1
        rank_loss2 = rank_image_loss2 + rank_text_loss2
        rank_loss3 = rank_image_loss3 + rank_text_loss3
        rank_loss  = rank_loss1+rank_loss2+rank_loss3
        return rank_loss
    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.context_embeds,
                    "lr": lr_prompt_learner_context,
                    "init_lr": lr_prompt_learner_context,
                    "name": "lr_prompt_learner_context",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.context_embeds)")
            try:
                freeze_param(self.module.prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_embeds,
                    "lr": lr_prompt_learner_ranks,
                    "init_lr": lr_prompt_learner_ranks,
                    "name": "lr_prompt_learner_ranks",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_embeds)")
            try:
                freeze_param(self.module.prompt_learner.rank_embeds)
            except AttributeError:
                pass

        if lr_image_encoder > 0 and self.module.image_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_encoder)")
            freeze_param(self.module.image_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls
