from functools import partial
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from lewm.training.forward import lejepa_forward
from lewm.training.transforms import get_column_normalizer, get_img_preprocessor


def build_training_manager(cfg):
    # Phase 1: Dataset + transforms
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    # Phase 2: Parameter inference + injection
    encoder = instantiate(cfg.wm.encoder)
    hidden_dim = encoder.config.hidden_size
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    with open_dict(cfg):
        cfg.wm.predictor.hidden_dim = hidden_dim
        cfg.wm.predictor.output_dim = hidden_dim
        cfg.wm.projector.input_dim = hidden_dim
        cfg.wm.pred_proj.input_dim = hidden_dim
        cfg.wm.action_encoder.input_dim = effective_act_dim
        if cfg.wm.decoder.get("enabled", False):
            cfg.wm.decoder.num_patches = (cfg.img_size // cfg.patch_size) ** 2
            cfg.wm.decoder.patch_size = cfg.patch_size

    # Phase 3: Module instantiation
    predictor = instantiate(cfg.wm.predictor)
    action_encoder = instantiate(cfg.wm.action_encoder)
    projector = instantiate(cfg.wm.projector)
    pred_proj = instantiate(cfg.wm.pred_proj)

    decoder = None
    if cfg.wm.decoder.get("enabled", False):
        decoder = instantiate(cfg.wm.decoder)

    world_model = instantiate(
        cfg.wm.world_model,
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        decoder=decoder,
    )

    sigreg = instantiate(cfg.loss.sigreg)
    optimizers = OmegaConf.to_container(cfg.optimizers, resolve=True)

    # Phase 4: Training assembly
    data_module = spt.data.DataModule(train=train, val=val)
    lightning_module = spt.Module(
        model=world_model,
        sigreg=sigreg,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as file_obj:
        OmegaConf.save(cfg, file_obj)

    callbacks_dict = instantiate(cfg.callbacks, _convert_="partial")

    if "model_checkpoint" in callbacks_dict:
        callbacks_dict["model_checkpoint"].dirpath = run_dir

    callbacks_list = list(callbacks_dict.values())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks_list,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=lightning_module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    return manager
