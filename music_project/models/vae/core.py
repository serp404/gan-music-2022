from music_project.models.vae.layers import *
from music_project.models.vae.base import BaseVAE


class MelVAE(BaseVAE):
    def __init__(self, input_size=(64, 15), latent_dim=32, is_featExtract=False,
                 n_convLayer=3, n_convChannel=[32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2],
                 n_fcLayer=1, n_fcChannel=[256], device=torch.device('cuda')):
        """
        Construction of VAE
        :param input_size: (n_channel, n_freqBand, n_contextWin);
                           assume a spectrogram input of size (n_freqBand, n_contextWin)
        :param latent_dim: the dimension of the latent vector
        :param is_featExtract: if True, output z as mu; otherwise, output z derived from reparameterization trick
        """
        super(MelVAE, self).__init__(input_size, latent_dim, is_featExtract, device)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.is_featExtract = is_featExtract

        self.n_freqBand, self.n_contextWin = input_size

        # Construct encoder and Gaussian layers
        self.encoder = spec_conv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)
        self.flat_size, self.encoder_outputSize = self._infer_flat_size()
        self.encoder_fc = fc(n_fcLayer, [self.flat_size, *n_fcChannel], activation='tanh', batchNorm=True)
        self.mu_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)
        self.logvar_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)

        # Construct decoder
        self.decoder_fc = fc(n_fcLayer + 1, [self.latent_dim, *n_fcChannel[::-1], self.flat_size],
                             activation='tanh', batchNorm=True)
        self.decoder = spec_deconv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)

    def _infer_flat_size(self):
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def encode(self, x):
        if len(x.shape) == 4:
            assert x.shape[1] == 1
            x = x.squeeze(1)

        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))
        mu = self.mu_fc(h2)
        logvar = self.logvar_fc(h2)
        mu, logvar, z = self._infer_latent(mu, logvar)

        return mu, logvar, z

    def decode(self, z):
        h = self.decoder_fc(z)
        x_recon = self.decoder(h.view(-1, *self.encoder_outputSize))
        return x_recon

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        # print(x_recon.size(), mu.size(), var.size(), z.size())
        return x_recon, mu, logvar, z


if __name__ == '__main__':
    device = torch.device('cpu')
    from music_project.models.vae.core import MelVAE
    length = 31 #33
    model = MelVAE(input_size=(80, length), device=device)
    model = model.to(device)
    x = torch.rand((16, 1, 80, length))
    out = model(x)
    out[0].shape
