import torch
import numpy as np
from typing import Callable
from templates import LitModel
import templates_latent
from functools import partial

# Numerically unstable, but probably OK for small tensors
def onnx_compatible_cumprod(t: torch.Tensor, dim: int=0):
    x = t                        # [1, 2]
    x = torch.log(x)             # [log(1), log(2)]
    x = torch.cumsum(x, dim=dim) # [log(1), log(1)+log(2)] = [log(1), log(1*2)]
    x = torch.exp(x)             # [e^log(1), e^log(1*2)] = [1, 1*2]
    return x

# Traceable sampler
class _DDIMSamplerTorch(torch.nn.Module):
    def __init__(self, conf, dtype=torch.float32, is_lat=False):
        super().__init__()

        # Compute alphas based on T_orig
        if is_lat:
            betas = conf._make_latent_diffusion_conf(conf.T).betas
        else:
            betas = conf._make_diffusion_conf(conf.T).betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(alphas, axis=0), dtype=dtype) # constant
        self.T_orig = conf.T # constant
        self.is_lat = is_lat

    # Sample for T iteration
    # Contains dynamic control flow => untraceable
    @torch.jit.unused
    def sample(self, T: torch.Tensor, x0: torch.Tensor, eval_model: Callable):
        params = self.forward(T) # tuple

        x = x0

        # The below loop over the ternsor will use static iteration count
        # => cannot trace
        n_iter = torch.ones(1).repeat(T).size(0)
        for t in torch.arange(0, n_iter, device=x0.device).flip(0).view(-1, 1):
            x = self.sample_incr(t, x, eval_model, *params)

        return x

    # Sample incrementally (single iteration)
    def sample_incr(
        self,
        t,
        x,
        eval_model: Callable,
        timestep_map,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
    ):
        eta = 0.0

        def _predict_eps_from_xstart(x_t, t, pred_xstart):
            num = _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
            denom = _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            return num / denom

        def _predict_xstart_from_eps(x_t, t, eps):
            assert x_t.shape == eps.shape
            a1 = _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape)
            a2 = _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            return (a1 * x_t - a2 * eps)

        def _extract_into_tensor(arr, timesteps, broadcast_shape):
            res = arr.float().to(device=timesteps.device)[timesteps]
            while len(res.shape) < len(broadcast_shape):
                res = res[..., None]
            return res.expand(broadcast_shape)

        map_tensor = timestep_map.to(t.device)
        model_forward = eval_model(x, map_tensor[t])
        model_output = model_forward.pred

        pred_xstart = _predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        if not self.is_lat:
            pred_xstart = pred_xstart.clamp(-1, 1)
        
        eps = _predict_eps_from_xstart(x, t, pred_xstart)
        alpha_bar = _extract_into_tensor(alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(alphas_cumprod_prev, t, x.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        
        # Equation 12.
        mean_pred = (pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        
        return mean_pred

    # Get params given number of inference steps T
    def forward(self, T: torch.Tensor):
        dtype = self.alphas_cumprod.dtype
        const_one = torch.tensor([1.0], dtype=dtype)

        timestep_map = torch.linspace(0, self.T_orig, torch.ones(1).repeat(T.clip(2, None)).size(0) + 1, dtype=torch.int64)[:-1]
        alphas_cumprod = self.alphas_cumprod[timestep_map]
        padded = torch.cat((const_one, alphas_cumprod), dim=0)
        betas = 1 - padded[1:] / padded[:-1]
        
        # Then compute alphas etc. (GaussianDiffusionBeatGans)
        alphas = 1.0 - betas
        alphas_cumprod = onnx_compatible_cumprod(alphas, dim=0) # ONNX has no cumprod...
        alphas_cumprod_prev = torch.cat((const_one, alphas_cumprod[:-1]))
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # jit.trace requires a constant container (tuple instead of list, NamedTuple instead of dict)
        return (
            timestep_map,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
        )

class DDIMSamplerLat(_DDIMSamplerTorch):
    def __init__(self, conf, dtype=torch.float32):
        super().__init__(conf, dtype, is_lat=True)

class DDIMSamplerImg(_DDIMSamplerTorch):
    def __init__(self, conf, dtype=torch.float32):
        super().__init__(conf, dtype, is_lat=False)

class DiffAEModel(torch.nn.Module):
    def __init__(self, dset):
        super().__init__()
        conf = getattr(templates_latent, f'{dset}_autoenc_latent')()
        conf.pretrain = None
        conf.fp16 = False
        conf.seed = None # will be set globally if not None
        self.name = conf.name

        model = LitModel(conf)
        assert isinstance(model, torch.nn.Module), 'Not a torch module!'
        state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        self.dset_lats = model.conds

        self.res = conf.img_size
        self.lat_sampl = DDIMSamplerLat(conf)
        self.img_sampl = DDIMSamplerImg(conf)
        self.lat_net = model.ema_model.latent_net
        self.img_net = model.ema_model
        self.norm_z = conf.latent_znormalize
        if self.norm_z:
            self.conds_std = model.conds_std
            self.conds_mean = model.conds_mean

    def lat_denorm(self, lat):
        if self.norm_z:
            lat = (lat * self.conds_std.to(lat.device)) + self.conds_mean.to(lat.device)
        return lat

    @property
    def dev_lat(self):
        return self.lat_net.layers[0].linear.weight.device
    
    @property
    def dev_img(self):
        return self.img_net.input_blocks[0][0].weight.device
    
    def forward(self, T_lat: torch.Tensor, T_img: torch.Tensor, x0_lat: torch.Tensor, x0_img: torch.Tensor):
        lats = self.sample_lat(T_lat, x0_lat)
        img = self.sample_img(T_img, x0_img, lats)
        return img

    @torch.jit.export
    def sample_lat(self, T_lat: torch.Tensor, x0_lat: torch.Tensor):
        lats = self.lat_sampl.sample(T_lat, x0_lat, lambda x, t: self.lat_net(x, t))
        return self.lat_denorm(lats)
    
    @torch.jit.export
    def sample_img(self, T_img: torch.Tensor, x0_img: torch.Tensor, lats: torch.Tensor):
        eval_fn = partial(self.img_net, cond=lats) # make lats accessible in jitted fwd wrapper
        return self.img_sampl.sample(T_img, x0_img, eval_fn)

    # Get params used in incremental image sampling
    @torch.jit.export
    def get_img_sampl_params(self, T_img: torch.Tensor):
        return self.img_sampl.forward(T_img)

    # Run image sampler one step forward
    @torch.jit.export
    def sample_img_incr(
        self,
        t,
        x, # accumulated image
        lats,
        # Below params from self.get_img_sampl_params()
        timestep_map,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
    ):
        eval_model = partial(self.img_net, cond=lats)
        return self.img_sampl.sample_incr(
            t, x, eval_model, timestep_map, alphas_cumprod,
            alphas_cumprod_prev, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)