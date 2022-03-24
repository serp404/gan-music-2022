import numpy as np
import torch
from music_project.vae_trainer.loss.vae_loss import vae_loss
from tqdm import tqdm
import wandb

class SpecVaeTrainer():
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, optimizer, device, mel_length, vocoder,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None):
        # super(SpecVaeTrainer, self).__init__(model, metrics, optimizer)
        self.model = model
        self.device = device
        self.mel_length = mel_length
        # self.metrics = metrics
        self.optimizer = optimizer
        self.data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.loss = vae_loss
        self.vocoder = vocoder

    def _forward_and_computeLoss(self, x, target):
        x_recon, mu, logvar, z = self.model(x)
        loss_recon, loss_kl = self.loss(mu, logvar, x_recon, target)
        loss = loss_recon + loss_kl
        return loss, loss_recon, loss_kl


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (wav, mel) in tqdm(enumerate(self.data_loader)):
            x = mel.type('torch.FloatTensor').to(self.device)
            x = x[..., :self.mel_length]

            self.optimizer.zero_grad()
            loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            wandb.log({'train_loss': loss.item()})
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()
            # total_metrics += self._eval_metrics(output, target)

            # if batch_idx % self.log_step == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * self.data_loader.batch_size,
            #         self.data_loader.n_samples,
            #         100.0 * batch_idx / len(self.data_loader),
            #         loss.item()))
            #     # TODO: visualize input/reconstructed spectrograms in TensorBoard
            #     # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kl': total_kl / len(self.data_loader)
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        # if self.do_validation:
        val_log_dict = self._valid_epoch(epoch)
        wandb.log(val_log_dict)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_recon = 0
        total_val_kl = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (wav, mel) in tqdm(enumerate(self.valid_data_loader)):
                x = mel.type('torch.FloatTensor').to(self.device)
                x = x[..., :self.mel_length]

                loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())
                wandb.log({'val_loss': loss.item()})
                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl += loss_kl.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            for i in range(min(5, x.shape[0])):
                output = self.model(x)[0]
                reconstructed_wav_input = self.vocoder.inference(x.to(self.device)).cpu()
                reconstructed_wav_output = self.vocoder.inference(output.to(self.device)).cpu()
                wandb.log({f'reconstructed_input_wav{i}': wandb.Audio(reconstructed_wav_input[i].detach().cpu().numpy(),
                                                                      sample_rate=22050),
                           f'reconstructed_output_wav{i}': wandb.Audio(reconstructed_wav_output[i].detach().cpu().numpy(),
                                                                      sample_rate=22050),
                           f"val_input_spec{i}": wandb.Image(x[i]), f"val_output_spec{i}": wandb.Image(output[i])})

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss_total': total_val_loss / len(self.valid_data_loader),
            'val_loss_recon': total_val_recon / len(self.valid_data_loader),
            'val_loss_kl': total_val_kl / len(self.valid_data_loader)
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }