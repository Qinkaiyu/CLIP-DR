import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from clipdr.models import MODELS
from clipdr.models.CLIPDR import CLIPDR
from clipdr.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .fds import FDS
from .resnet import resnet50

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
        self.FDS = FDS(feature_dim = 5,bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9)
        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.list_logits = []
        self.list_y = []
        self.training_step_outputs = []

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()


    def test_step(self, batch, batch_idx):
        x, y = batch
        our_logits, image_features, text_features = self.module(x)
        model = self.module
        # Initialize Simple / Smooth FullGrad objects

        metrics_exp = self.compute_per_example_metrics(our_logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(our_logits, y, "max")

        return {**metrics_exp, **metrics_max}

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


        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)
        auc_ovo = roc_auc_score(y.cpu().numpy(), probs.cpu().detach().numpy(), average='macro', multi_class='ovo',labels=[0, 1, 2,3,4])
        #auc_ovo = auc_ovo*len(y)
        auc_ovo = torch.tensor(auc_ovo)
        if torch.isnan(auc_ovo):
            # 处理NaN值的情况，例如赋予一个默认值或者执行其他的操作
            print("Encountered NaN in test_exp_DGDR_auc_metric. Handling NaN...")
            auc_ovo = torch.tensor(1) # 用默认值替代NaN，你需要根据具体情况选择合适的默认值
        #DGDR_acc = accuracy_score(y.cpu().numpy(), predict_y.cpu().numpy())

        f1 = f1_score(y.cpu().numpy(), torch.round(predict_y).cpu().detach().numpy(), average='macro')
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
