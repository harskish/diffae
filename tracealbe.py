from turtle import forward
import torch
from typing import Callable
from templates import LitModel
import templates_latent
from config import TrainMode

# Traceable sampler
class _DDIMSamplerTorch(torch.nn.Module):
    def __init__(self, conf, dtype=torch.float32, is_lat=False):
        super().__init__()

        # Compute alphas based on T_orig
        if is_lat:
            betas = conf._make_latent_diffusion_conf(conf.T).betas
        else:
            betas = conf._make_diffusion_conf(conf.T).betas
        alphas = 1.0 - torch.tensor(betas, dtype=dtype)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0) # constant
        self.T_orig = conf.T # constant
        self.is_lat = is_lat

    # Sample for T iterations
    def sample(self, T: torch.Tensor, x0: torch.Tensor, eval_model: Callable):
        params = self.forward(T)

        x = x0
        n_iter = torch.ones(1).repeat(T).size(0)
        for t in torch.arange(0, n_iter, device=x0.device).flip(0).view(-1, 1):
            x = self.sample_incr(t, x, eval_model, **params)

        return x

    # Sample incrementally (single iteration)
    def sample_incr(
        self,
        t,
        x,
        eval_model: Callable,
        timestep_map,
        posterior_variance,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        betas,
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

        model_variance = torch.cat((posterior_variance[1].view(-1), betas[1:]))
        model_log_variance = torch.log(torch.cat((posterior_variance[1].view(-1), betas[1:])))
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
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
        const_zero = torch.tensor([0.0], dtype=dtype)
        const_one = torch.tensor([1.0], dtype=dtype)

        timestep_map = torch.linspace(0, self.T_orig, torch.ones(1).repeat(T).size(0) + 1, dtype=torch.int64)[:-1]
        alphas_cumprod = self.alphas_cumprod[timestep_map]
        padded = torch.cat((const_one, alphas_cumprod), dim=0)
        betas = 1 - padded[1:] / padded[:-1]
        
        # Then compute alphas etc. (GaussianDiffusionBeatGans)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((const_one, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.cat((alphas_cumprod[1:], const_zero))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = torch.log(
            torch.cat((posterior_variance[1].view(-1), posterior_variance[1:])))
        posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        return {
            'timestep_map': timestep_map,
            'posterior_variance': posterior_variance,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
            'sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod,
            'betas': betas,
        }

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
        self.name = conf.name

        model = LitModel(conf)
        assert isinstance(model, torch.nn.Module), 'Not a torch module!'
        state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)

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
        return self.img_sampl.sample(T_img, x0_img, lambda x, t: self.img_net(x, t, cond=lats))

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
        posterior_variance,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        betas
    ):
        eval_model = lambda x, t: self.img_net(x, t, cond=lats)
        return self.img_sampl.sample_incr(
            t, x, eval_model, timestep_map, posterior_variance, alphas_cumprod,
            alphas_cumprod_prev, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, betas)