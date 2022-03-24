import os
import argparse
import warnings
import random
import wandb

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from music_project.common.melspec import MelSpectrogram, MelSpectrogramConfig
from music_project.common.dataset import MusicDataset
from music_project.models.hifi import HiFiGenerator, \
    MultiPeriodDiscriminator, MultiScaleDiscriminator
from music_project.common.utils import get_grad_norm, \
    clip_gradients, traverse_config
from music_project.common.collator import split_collate_fn
from music_project.common.utils import init_scheduler
from music_project.config import TaskConfig

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = TaskConfig()
MEL_CONFIG = MelSpectrogramConfig()

SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def main(args):
    pargs = args.parse_args()
    resume_gen = pargs.resume_gen
    resume_dis = pargs.resume_dis
    device_opt = pargs.device

    if device_opt is not None:
        DEVICE = torch.device(device_opt)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_EPOCHS = CONFIG.n_epochs
    BATCH_SIZE = CONFIG.dataloaders_params["batch_size"]

    # Initialize wandb logs
    wandb.login()
    run = wandb.init(
        project="music_project",
        entity="serp404",
        config=traverse_config(CONFIG)
    )

    save_path = os.path.join(CONFIG.save_dir, run.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Preparing dataset and dataloaders
    data_params = CONFIG.dataloaders_params
    dataset = MusicDataset(data_path="./data")
    total_len = len(dataset)

    train_len = int(data_params["train_size"] * total_len)
    val_len = total_len - train_len

    train_part, val_part = torch.utils.data.random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_part, batch_size=BATCH_SIZE, collate_fn=split_collate_fn,
        shuffle=True, num_workers=data_params["num_workers"], pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_part, batch_size=BATCH_SIZE, collate_fn=split_collate_fn,
        shuffle=True, num_workers=data_params["num_workers"], pin_memory=True
    )

    gen = HiFiGenerator(**CONFIG.gen_params).to(DEVICE)

    # Load generator weights if needed
    if resume_gen is not None:
        gen.load_state_dict(torch.load(resume_gen))

    dis = nn.ModuleList([
        MultiPeriodDiscriminator(**CONFIG.mpd_params),
        MultiScaleDiscriminator(**CONFIG.msd_params)
    ]).to(DEVICE)

    # Load discriminators weights if needed
    if resume_dis is not None:
        dis.load_state_dict(torch.load(resume_dis))

    optimizer_gen = getattr(torch.optim, CONFIG.optimizer_gen)(
        gen.parameters(), **CONFIG.optimizer_gen_params
    )

    scheduler_gen = init_scheduler(
        CONFIG.scheduler_gen, CONFIG.scheduler_gen_params, optimizer_gen
    )

    optimizer_dis = getattr(torch.optim, CONFIG.optimizer_dis)(
        dis.parameters(), **CONFIG.optimizer_dis_params
    )

    scheduler_dis = init_scheduler(
        CONFIG.scheduler_dis, CONFIG.scheduler_dis_params, optimizer_dis
    )

    mel_featurizer = MelSpectrogram(MEL_CONFIG)
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()

    # Mini-batch for tracking in wandb
    example_wavs, example_mels = next(iter(val_loader))
    example_wavs = example_wavs[:CONFIG.examples_cnt]
    example_mels = example_mels[:CONFIG.examples_cnt]

    real_audios = [
        wandb.Audio(wav.numpy(), sample_rate=MEL_CONFIG.sr)
        for wav in example_wavs
    ]

    real_melspecs = [
        wandb.Image(mel.numpy())
        for mel in example_mels
    ]

    for epoch in range(N_EPOCHS):
        if scheduler_gen is not None:
            scheduler_gen.step()

        if scheduler_dis is not None:
            scheduler_dis.step()

        # Training loop
        gen.train()
        dis.train()
        train_losses_gen = []
        train_losses_dis = []
        train_grads_gen = []
        train_grads_dis = []
        for i, batch in tqdm(
            enumerate(train_loader), desc=f"Training epoch {epoch}",
            total=CONFIG.n_train_iters
        ):

            mels_real = batch[1][:BATCH_SIZE].to(DEVICE)
            wavs_real = batch[0][:BATCH_SIZE].unsqueeze(dim=1).to(DEVICE)
            n_samples = wavs_real.shape[0]

            wavs_fake = gen(mels_real)
            mels_fake = mel_featurizer(
                wavs_fake.squeeze(dim=1).cpu()
            ).to(DEVICE)

            real_labels = torch.ones(n_samples, device=DEVICE)
            fake_labels = torch.zeros(n_samples, device=DEVICE)

            # discriminator
            loss_dis = 0.
            for d in dis:
                preds_real, _ = d(wavs_real)
                loss_dis += l2_criterion(preds_real, real_labels)
                preds_fake, _ = d(wavs_fake)
                loss_dis += l2_criterion(preds_fake.detach(), fake_labels)

            optimizer_dis.zero_grad()
            loss_dis.backward()
            clip_gradients(dis.parameters(), CONFIG.grad_clip)
            train_grads_dis.append(get_grad_norm(dis))
            optimizer_dis.step()

            # generator
            max_len = min(mels_real.shape[-1], mels_fake.shape[-1])
            loss_gen = 45. * l1_criterion(
                mels_real[:, :, :max_len],
                mels_fake[:, :, :max_len]
            )
            for d in dis:
                preds_real, fmaps_real = d(wavs_real)
                preds_fake, fmaps_fake = d(wavs_fake)
                loss_gen += l2_criterion(preds_fake, real_labels)
                for fms_r, fms_f in zip(fmaps_real, fmaps_fake):
                    for fm_r, fm_f in zip(fms_r, fms_f):
                        max_len = min(fm_r.shape[-1], fm_f.shape[-1])
                        loss_gen += 2. * l2_criterion(
                            fm_r[:, :, :max_len],
                            fm_f[:, :, :max_len]
                        )

            optimizer_gen.zero_grad()
            loss_gen.backward()
            clip_gradients(gen.parameters(), CONFIG.grad_clip)
            train_grads_gen.append(get_grad_norm(gen))
            optimizer_gen.step()

            train_losses_dis.append(loss_dis.item())
            train_losses_gen.append(loss_gen.item())

            if (i+1) % CONFIG.n_train_iters == 0:
                break

        # Validation loop
        gen.eval()
        dis.eval()
        val_losses_gen = []
        val_losses_dis = []
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(val_loader), desc=f"Validating epoch {epoch}",
                total=CONFIG.n_val_iters
            ):
                mels_real = batch[1][:BATCH_SIZE].to(DEVICE)
                wavs_real = batch[0][:BATCH_SIZE].unsqueeze(dim=1).to(DEVICE)
                n_samples = wavs_real.shape[0]

                wavs_fake = gen(mels_real)
                mels_fake = mel_featurizer(
                    wavs_fake.squeeze(dim=1).cpu()
                ).to(DEVICE)

                real_labels = torch.ones(n_samples, device=DEVICE)
                fake_labels = torch.zeros(n_samples, device=DEVICE)

                # discriminator
                loss_dis = 0.
                for d in dis:
                    preds_real, _ = d(wavs_real)
                    loss_dis += l2_criterion(preds_real, real_labels)
                    preds_fake, _ = d(wavs_fake)
                    loss_dis += l2_criterion(preds_fake, fake_labels)

                # generator
                max_len = min(mels_real.shape[-1], mels_fake.shape[-1])
                loss_gen = 45. * l1_criterion(
                    mels_real[:, :, :max_len],
                    mels_fake[:, :, :max_len]
                )
                for d in dis:
                    preds_real, fmaps_real = d(wavs_real)
                    preds_fake, fmaps_fake = d(wavs_fake)
                    loss_gen += l2_criterion(preds_fake, real_labels)
                    for fms_r, fms_f in zip(fmaps_real, fmaps_fake):
                        for fm_r, fm_f in zip(fms_r, fms_f):
                            max_len = min(fm_r.shape[-1], fm_f.shape[-1])
                            loss_gen += 2. * l2_criterion(
                                fm_r[:, :, :max_len],
                                fm_f[:, :, :max_len]
                            )

                val_losses_dis.append(loss_dis.item())
                val_losses_gen.append(loss_gen.item())

                if (i+1) % CONFIG.n_val_iters == 0:
                    break

        mels_real = example_mels.to(DEVICE)
        with torch.no_grad():
            predicted_wavs = gen(mels_real).squeeze(dim=1)
        predicted_mels = mel_featurizer(wavs_fake.cpu()).to(DEVICE)

        predicted_audios = [
            wandb.Audio(
                predicted_wavs[i].cpu().numpy(),
                sample_rate=MEL_CONFIG.sr
            ) for i in range(CONFIG.examples_cnt)
        ]

        predicted_melspecs = [
            wandb.Image(
                predicted_mels[i].cpu().numpy()
            ) for i in range(CONFIG.examples_cnt)
        ]

        wandb.log(
            {
                "epoch": epoch,
                "train_loss_gen": np.mean(train_losses_gen),
                "train_loss_dis": np.mean(train_losses_dis),
                "val_loss_gen": np.mean(val_losses_gen),
                "val_loss_dis": np.mean(val_losses_dis),
                "grad_norm_gen": np.mean(train_grads_gen),
                "grad_norm_dis": np.mean(train_grads_dis),
                "lr_gen": optimizer_gen.param_groups[0]['lr'],
                "lr_dis": optimizer_dis.param_groups[0]['lr'],
                "val_predicted_audios": predicted_audios,
                "val_predicted_melspecs": predicted_melspecs,
                "val_real_audios": real_audios,
                "val_real_melspecs": real_melspecs
            }
        )

        if epoch % CONFIG.save_period == 0:
            torch.save(
                gen.state_dict(),
                os.path.join(save_path, f"gen_e{epoch}.pth")
            )

            torch.save(
                dis.state_dict(),
                os.path.join(save_path, f"dis_e{epoch}.pth")
            )

    torch.save(
        gen.state_dict(),
        os.path.join(save_path, "gen_final.pth")
    )

    torch.save(
        dis.state_dict(),
        os.path.join(save_path, "dis_final.pth")
    )

    run.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="HifiGan train")
    args.add_argument(
        "-g",
        "--resume_gen",
        default=None,
        type=str,
        help="path to latest generator checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--resume_dis",
        default=None,
        type=str,
        help="path to latest discriminator checkpoint (default: None)",
    )
    args.add_argument(
        "-D",
        "--device",
        default=None,
        type=str,
        help="cpu of cuda (default: cuda if possible else cpu)",
    )

    main(args)
