import torch
import transformers


def init_normal_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def normalize_simple_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.utils.weight_norm(m)


def normalize_spectral_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.utils.spectral_norm(m)


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def clip_gradients(params, clip_value):
    if clip_value is not None:
        torch.nn.utils.clip_grad_norm_(params, clip_value)


def traverse_config(config):
    return {
        attr: getattr(config, attr) for attr in dir(config)
        if not attr.startswith("__")
    }


def wav_splitter(wav, featurizer, max_len):
    for wav_part in torch.split(wav, max_len, dim=1):
        if wav_part.shape[-1] >= max_len // 10:
            mels = featurizer(wav_part)
            yield wav_part, mels


def init_scheduler(scheduler_type, scheduler_params, optimizer):
    if scheduler_type is not None:
        if scheduler_type in dir(torch.optim.lr_scheduler):
            return getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer, **scheduler_params
            )
        elif scheduler_type in dir(transformers.optimization):
            return getattr(transformers.optimization, scheduler_type)(
                optimizer, **scheduler_params
            )
        else:
            raise ModuleNotFoundError(f"Unknown scheduler '{scheduler_type}'")
