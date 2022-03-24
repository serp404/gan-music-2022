import typing as tp


class TaskConfig:
    # Train options
    n_epochs: int = 100
    n_train_iters: int = 200
    n_val_iters: int = 20
    max_wav_len: int = 8192
    mel_length: int = 31
    dataloaders_params: tp.Dict[str, tp.Any] = {
        "batch_size": 32,
        "train_size": 0.8,
        "num_workers": 4
    }

    # Model options
    gen_params: tp.Dict[str, tp.Any] = {
        "channels_u": 256,
        "kernels_u": (16, 16, 8),
        "kernels_r": (3, 5, 7),
        "dilations_r": ((1, 2), (2, 6), (3, 12)),
        "slope": 0.1
    }

    mpd_params: tp.Dict[str, tp.Any] = {}
    msd_params: tp.Dict[str, tp.Any] = {}

    # Optimization options
    grad_clip: tp.Optional[float] = 3.
    optimizer_gen: str = "Adam"
    optimizer_gen_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0002,
        "betas": (0.8, 0.99),
        "weight_decay": 0.01
    }

    scheduler_gen: tp.Optional[str] = "CosineAnnealingLR"
    scheduler_gen_params: tp.Dict[str, tp.Any] = {"T_max": 100}

    optimizer_dis: str = "Adam"
    optimizer_dis_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0001,
        "betas": (0.8, 0.99),
        "weight_decay": 0.01
    }

    scheduler_dis: tp.Optional[str] = "CosineAnnealingLR"
    scheduler_dis_params: tp.Dict[str, tp.Any] = {"T_max": 100}

    # Checkpoints
    save_period: int = 3
    save_dir: str = "./log"
    examples_cnt: int = 5
