from typing import Any, Dict, List, Tuple

import os
import hydra
import torch
import rootutils
import subprocess
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.checkpoints

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.module)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")

    # for predictions use trainer.predict(...)
    for ckpt_cfg in cfg.checkpoints:
        ckpt_path = os.path.join(cfg.paths.log_dir, ckpt_cfg.name, 'runs', ckpt_cfg.date, 'checkpoints', ckpt_cfg.ckpt)
        ckpt = torch.load(ckpt_path, weights_only=False)
        # If the model was compiled, we need to strip the "_orig_mod" prefix from the dict keys
        renamed_ckpt = {} 
        for k, v in ckpt["state_dict"].items():
            new_k = k.replace("_orig_mod.", "")
            renamed_ckpt[new_k] = v
        model.load_state_dict(renamed_ckpt)
        predictions = trainer.predict(model=model, datamodule=datamodule, return_predictions=True)
        predictions = torch.cat(predictions, axis=0)
    
        out_dir = os.path.join(cfg.paths.prediction_dir, ckpt_cfg.name, ckpt_cfg.date)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'predictions.pt')
        torch.save(predictions, out_path)
        
        # Send to evaluation server
        subprocess.call([
            "rsync", out_path, f"172.22.11.44::eval_server/{cfg.team_name}/{ckpt_cfg.name}/"
        ])
        emissions_path = os.path.join(cfg.paths.codecarbon_dir, 'emissions.csv')
        subprocess.call([
            "rsync", emissions_path, f"172.22.11.44::eval_server/{cfg.team_name}/"
        ])

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    torch.set_float32_matmul_precision('high') # To use Tensor cores
    evaluate(cfg)


if __name__ == "__main__":
    main()