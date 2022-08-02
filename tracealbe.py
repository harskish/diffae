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

        # The size of timestep_map will be static in onnx export...
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
    def __init__(self, dset, dev_lat='cpu', dev_img='cpu', lat_fused=False, img_fused=False):
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
        self.lat_sampl = DDIMSamplerLat(conf).to(dev_lat)
        self.img_sampl = DDIMSamplerImg(conf).to(dev_img)
        self.img_net = model.ema_model.to(dev_img) # sets both
        self.lat_net = model.ema_model.latent_net.to(dev_lat) # overrides
        self.dev_lat = torch.device(dev_lat)
        self.dev_img = torch.device(dev_img)

        # Fused mode: recompute ddim sampler params for every step
        # Less memory traffic at the cost of recomputing stuff
        self.lat_fused = lat_fused
        self.img_fused = img_fused

        self.norm_z = conf.latent_znormalize
        if self.norm_z:
            self.conds_std = model.conds_std
            self.conds_mean = model.conds_mean

    def lat_denorm(self, lat):
        if self.norm_z:
            lat = (lat * self.conds_std.to(lat.device)) + self.conds_mean.to(lat.device)
        return lat
    
    @torch.jit.unused # dynamic control flow
    def forward(self, T_lat: torch.Tensor, T_img: torch.Tensor, x0_lat: torch.Tensor, x0_img: torch.Tensor):
        lats = self.sample_lat_loop(T_lat, x0_lat)
        img = self.sample_img_loop(T_img, x0_img, lats.to(self.dev_img))
        return img

    @torch.jit.unused # dynamic control flow
    def sample_lat_loop(self, T_lat: torch.Tensor, x0_lat: torch.Tensor):
        params = None if self.lat_fused else self.lat_sampl(T_lat)
        
        x = x0_lat
        for i in range(T_lat.item()):
            t = T_lat - i - 1
            if self.lat_fused:
                x = self.sample_lat_incr_fused(T_lat, t, x)
            else:
                x = self.sample_lat_incr(t, x, *params)
        
        return self.lat_denorm(x)
    
    @torch.jit.unused # dynamic control flow
    def sample_img_loop(self, T_img: torch.Tensor, x0_img: torch.Tensor, lats: torch.Tensor):
        params = None if self.img_fused else self.img_sampl(T_img)
        
        x = x0_img
        for i in range(T_img.item()):
            t = T_img - i - 1
            if self.img_fused:
                x = self.sample_img_incr_fused(T_img, t, x, lats)
            else:
                x = self.sample_img_incr(t, x, lats, *params)

        return x

    # Get params used in incremental latent sampling
    @torch.jit.export
    def get_lat_sampl_params(self, T_lat: torch.Tensor):
        assert not self.lat_fused, 'Should not query in fused mode'
        return self.lat_sampl.forward(T_lat)
    
    # Get params used in incremental image sampling
    @torch.jit.export
    def get_img_sampl_params(self, T_img: torch.Tensor):
        assert not self.img_fused, 'Should not query in fused mode'
        return self.img_sampl.forward(T_img)

    # Run latent sampler one step forward
    @torch.jit.export
    def sample_lat_incr(
        self,
        t,
        x, # accumulated latent
        # Below params from self.get_lat_sampl_params()
        timestep_map,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
    ):
        assert not self.lat_fused, 'Fused mode is on, use sample_lat_incr_fused() instead'
        return self.lat_sampl.sample_incr(
            t, x, self.lat_net, timestep_map, alphas_cumprod,
            alphas_cumprod_prev, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)
    
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
        assert not self.img_fused, 'Fused mode is on, use sample_img_incr_fused() instead'
        eval_model = partial(self.img_net, cond=lats)
        return self.img_sampl.sample_incr(
            t, x, eval_model, timestep_map, alphas_cumprod,
            alphas_cumprod_prev, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)

    # Fused methods: recomputes alphas for every step
    # Upside: no separate init-params file, less memory traffic
    # Downside: recomputing stuff every iteration
    
    @torch.jit.export
    def sample_lat_incr_fused(
        self,
        T: torch.Tensor,   # total number of steps
        t: torch.Tensor,   # current step, T-1 -> 0
        x: torch.Tensor,   # accumulated intermediate result
    ):
        params = self.lat_sampl.forward(T) # <- if this is small, then overhead is minimal
        return self.lat_sampl.sample_incr(t, x, self.lat_net, *params)
    
    @torch.jit.export
    def sample_img_incr_fused(
        self,
        T: torch.Tensor,   # total number of steps
        t: torch.Tensor,   # current step, T-1 -> 0
        x: torch.Tensor,   # accumulated intermediate result
        lats: torch.Tensor # conditioning (latent) vector
    ):
        params = self.img_sampl.forward(T) # <- if this is small, then overhead is minimal
        return self.img_sampl.sample_incr(t, x, partial(self.img_net, cond=lats), *params)