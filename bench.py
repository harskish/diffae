# Benchmark DiffAE performance on Apple M2 using different backends
# - Apache TVM (w/ optimizations)
# - coremltools (using NE)
# - Onnxruntime (CoreML, w/ NE?) - only C++?

from getopt import getopt
from random import Random
import torch
import numpy as np
from numpy.random import RandomState
from model.unet_autoenc import AutoencReturn
from templates import LitModel
from diffusion.base import _extract_into_tensor
import templates_latent
from config import TrainMode
from PIL import Image
from tqdm import trange
import time
import re

torch.autograd.set_grad_enabled(False)

# Repr monkey patch
_orig = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: _orig(t.cpu())

def show(t):
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    if np.issubdtype(t.dtype, np.floating):
        if t.min() < 0.1:
            t = 0.5*(t + 1)
        t = np.uint8(255*np.clip(t, 0, 1))
    t = t.squeeze()
    if t.ndim == 2:
        t = np.expand_dims(t, -1)
    if t.shape[0] in [1, 3, 4]: # CHW => HWC
        t = np.transpose(t, (1, 2, 0))
    Image.fromarray(t).show()

def get_build_opt(name: str):
    pattern = r'(-D)?{}=?([^\s,]*)?'.format(name)
    matches = re.findall(pattern, torch.__config__.show(), flags=re.IGNORECASE)
    vals = [m[1] for m in matches if m != ('')] # only -DFOO or FOO=BAR
    
    if not vals:
        return None # not found
    elif vals[0] == '':
        return True # defined
    else:
        return vals[0] # set to value

def is_set(name: str):
    val = str(get_build_opt(name)).upper()
    return not val in ['NONE', 'FALSE', 'OFF', '0']

def get_samplers(conf, t_img=10, t_lat=10):
    conf_img = conf._make_diffusion_conf(conf.T)
    conf_img.use_timesteps = np.linspace(0, conf.T, max(2, t_img) + 1, dtype=np.int32).tolist()[:-1]
    sampl = conf_img.make_sampler()

    conf_lat = conf._make_latent_diffusion_conf(conf.T)
    conf_lat.use_timesteps = np.linspace(0, conf.T, max(2, t_lat) + 1, dtype=np.int32).tolist()[:-1]
    lat_sampl = conf_lat.make_sampler()

    return (sampl, lat_sampl)

def run(model, steps=10, B=1, verbose=True, seed=None):
    seed = seed or np.random.randint(0, 1<<32-1)
    steps_lat = 10
    sampl, lat_sampl = get_samplers(model.conf, steps, steps_lat)
    ema_model = model.ema_model
    conf = model.conf
    ran_fun = trange if verbose else range
    dev_lat = model.ema_model.latent_net.layers[0].linear.weight.device
    dev_img = model.ema_model.input_blocks[0][0].weight.device
    
    # Warmup
    _x1 = torch.randn((B, 512))
    _x2 = torch.randn((B, 3, model.conf.img_size, model.conf.img_size))
    _t = torch.tensor([9] * B)
    for _ in range(3):
        _ = model.ema_model.latent_net(_x1.to(dev_lat), _t.to(dev_lat))
        _ = model.ema_model(_x2.to(dev_img), _t.to(dev_img), cond=_x1.to(dev_img))

    t0 = time.time()
    latent_noise = torch.tensor(RandomState(seed).randn(B, 512), dtype=torch.float32).to(dev_lat)
    
    # diffusion.base.GaussianDiffusionBeatGANs::ddim_sample()
    #   GaussianDiffusionBeatGANs::p_mean_variance()
    #     model.forward(x=x, t=t)
    lats = lat_sampl.sample(
        model=ema_model.latent_net,
        noise=latent_noise,
        clip_denoised=conf.latent_clip_sample,
        progress=False,
    ).to(dev_img)
    t1 = time.time()

    # Avoid double normalization
    if conf.latent_znormalize:
        lats = model.denormalize(lats)
    
    # Initial value: spaial noise
    intermed = torch.tensor(RandomState(seed).randn(B, 3, conf.img_size, conf.img_size), dtype=torch.float32).to(dev_img)

    for i in ran_fun(steps):
        t = torch.tensor([steps - i - 1] * B, device=dev_img, requires_grad=False)
        
        def _predict_eps_from_xstart(x_t, t, pred_xstart):
            num = _extract_into_tensor(sampl.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
            denom = _extract_into_tensor(sampl.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            return num / denom

        x = intermed
        eta = 0.0
        
        #################################
        # Run model with scaled timestamp
        #################################
        map_tensor = torch.tensor(sampl.timestep_map, device=t.device, dtype=t.dtype)
        model_forward = ema_model.forward(x=x, t=map_tensor[t], x_start=None, cond=lats)
        model_output = model_forward.pred

        model_variance = np.append(sampl.posterior_variance[1], sampl.betas[1:])
        model_log_variance = np.log(np.append(sampl.posterior_variance[1], sampl.betas[1:]))
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        pred_xstart = sampl._predict_xstart_from_eps(x_t=x, t=t, eps=model_output).clamp(-1, 1)
        
        eps = _predict_eps_from_xstart(x, t, pred_xstart)
        alpha_bar = _extract_into_tensor(sampl.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(sampl.alphas_cumprod_prev, t, x.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        
        # Equation 12.
        mean_pred = (pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        intermed = mean_pred

    t2 = time.time()
    
    it_lat = (t1 - t0) / steps_lat
    it_img = (t2 - t1) / steps
    it_tot = (t2 - t0) / steps

    if verbose:
        desc = ' [TR]' if model.traced else ''
        print(f'{dev_img.type}{desc} - Lat: {1/it_lat:.2f}it/s, img: {1/it_img:.2f}it/s, tot: {1/it_tot:.2f}it/s')

    return intermed

def model_torch(dev, dset):
    conf = getattr(templates_latent, f'{dset}_autoenc_latent')()
    conf.seed = None
    conf.pretrain = None
    conf.fp16 = False

    assert conf.train_mode == TrainMode.latent_diffusion
    assert conf.model_type.has_autoenc()

    model = LitModel(conf)
    assert isinstance(model, torch.nn.Module), 'Not a torch module!'
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    setattr(model, 'traced', False)
    
    return model.to(dev)

# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
def model_torch_traced(dev, dset, B=1):
    model = model_torch(dev, dset)
    
    if is_set('USE_MKLDNN') and is_set('BUILD_ONEDNN_GRAPH'):
        torch.jit.enable_onednn_fusion(True)

    x = torch.randn((B, 512)).to(dev)
    t = torch.tensor([9] * 1, device=dev, requires_grad=False) # unscaled

    # Latent net
    fwd_fun = lambda x, t: model.ema_model.latent_net(x, t).pred
    out_ref = fwd_fun(x, t)
    jit_lat = torch.jit.trace(fwd_fun, (x, t), check_trace=False)
    jit_lat.save(f'{dset}_lat.pt')
    assert torch.allclose(out_ref.cpu(), jit_lat(x, t).cpu()), 'Lat-net outputs differ'
    
    # Export ONNX
    torch.onnx.export(jit_lat,                      # model being run
                      (x, t),                       # model input (or a tuple for multiple inputs)
                      f'{dset}_lat.onnx',           # where to save the model (can be a file or file-like object)
                      input_names = ['z', 't'],     # the model's input names
                      output_names = ['output'])    # the model's output names
    
    # Replace with jitted
    def new_fwd_lat(x, t, **kwargs):
        return templates_latent.LatentNetReturn(jit_lat(x, t))
    model.ema_model.latent_net.forward = new_fwd_lat

    # Image diffusion
    cond = torch.randn((B, 512), device=dev)
    x = torch.randn((B, 3, model.conf.img_size, model.conf.img_size)).to(dev)
    fwd_fun = lambda x, t, cond: model.ema_model(x, t, cond=cond, x_start=None).pred
    out_ref = fwd_fun(x, t, cond)
    jit_img = torch.jit.trace(fwd_fun, (x, t, cond), check_trace=False)
    jit_img.save(f'{dset}_img.pt')
    #assert torch.allclose(out_ref.cpu(), jit_img(x, t, cond).cpu()), 'Img-net outputs differ'

    torch.onnx.export(jit_img,                          # model being run
                      (x, t, cond),                     # model input (or a tuple for multiple inputs)
                      f'{dset}_img.onnx',               # where to save the model (can be a file or file-like object)
                      input_names = ['z', 't', 'cond'], # the model's input names
                      output_names = ['output'])        # the model's output names

    # Replace with jitted
    def new_fwd_img(x, t, cond=None, **kwargs):
        assert not any(kwargs.values()), 'Unsupported kwargs provided'
        ret = jit_img(x, t, cond=cond)
        return AutoencReturn(pred=ret, cond=cond)
    model.ema_model.forward = new_fwd_img

    setattr(model, 'traced', True)
    return model

# M2 optimal: mix of traced CPU and GPU
def get_model_m2_optimal(dset):
    # Image diffusion fastest on traced MPS
    model = model_torch_traced('mps', dset)
    
    # Traced latent net fastest on CPU
    model.ema_model.latent_net = model_torch_traced('cpu', dset).ema_model.latent_net

    return model

import torch.nn as nn

class DiffAE(nn.Module):
    def __init__(self, steps=10, steps_lat=None, traced=False):
        super().__init__()
        model_constr = model_torch_traced if traced else model_torch
        model = model_constr('mps', 'ffhq256')
        self.model = model.ema_model
        self.lat_model = self.model.latent_net
        self.conf = model.conf
        self.sampl = self.lat_sampl = None
        self.init_lat_sampler(steps_lat or steps)
        self.init_img_sampler(steps)
        self.register_buffer('conds_std', model.conds_std)
        self.register_buffer('conds_mean', model.conds_mean)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_img_sampler(self, steps=10):
        conf_img = self.conf._make_diffusion_conf(self.conf.T)
        conf_img.use_timesteps = np.linspace(0, self.conf.T, max(2, steps) + 1, dtype=np.int32).tolist()[:-1]
        self.sampl = conf_img.make_sampler()

    def init_lat_sampler(self, steps=10):
        conf_lat = self.conf._make_latent_diffusion_conf(self.conf.T)
        conf_lat.use_timesteps = np.linspace(0, self.conf.T, max(2, steps) + 1, dtype=np.int32).tolist()[:-1]
        self.lat_sampl = conf_lat.make_sampler()

    def denorm_z(self, cond):
        cond = (cond * self.conds_std.to(cond.device)) + self.conds_mean.to(cond.device)
        return cond

    def run_lat_model(self, z, steps=10):
        self.init_lat_sampler(steps)
        lats = self.lat_sampl.sample(
            model=self.lat_model,
            noise=z,
            clip_denoised=self.conf.latent_clip_sample,
            progress=False
        )

        return lats
    
    def forward(self, z, steps, steps_lat=None, B=1):
        # Get conditioning ('latent vector')
        cond = self.run_lat_model(z, steps_lat or steps)
        
        # Avoid double normalization
        if self.conf.latent_znormalize:
            cond = self.denorm_z(cond)
        
        # Initial value: spaial noise
        intermed = torch.randn((B, 3, self.conf.img_size, self.conf.img_size)).to(self.model.device)
        model_kwargs = {'x_start': None, 'cond': cond}
        self.init_img_sampler(steps)
        
        sched = torch.arange(steps, device=self.model.device).flip(0).view(-1, 1)
        for t in sched:
            ret = self.sampl.ddim_sample(
                self.model,
                intermed,
                t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=model_kwargs,
                eta=0.0,
            )
            intermed = ret['sample']
        
        return intermed

# Partial trace
#z = torch.randn(1, 512).to('mps')
#test = DiffAE(traced=True)
#img = test.forward(z, 10, 10)
#show(img)

# FULL TRACE
# z = torch.randn(1, 512).to('mps')
# model = DiffAE(traced=False)
# tl = ti = 10
# full_trace = torch.jit.trace(lambda z: model(z, ti, tl), (z), check_trace=False) # step counts fixed
# full_trace.save(f'ffhq256_{ti}i_{tl}l_full.pt')
# show(full_trace(z))

# Still missing: script + trace
# Also: *tracing* can produce ScriptModule (which can be optimized!)
# TODO

from functools import partial
CONFIGS = {
    'cuda': partial(model_torch, 'cuda'),
    'cuda_traced': partial(model_torch_traced, 'cuda'),
    'cpu': partial(model_torch, 'cpu'),
    'cpu_traced': partial(model_torch_traced, 'cpu'),
    'mps': partial(model_torch, 'mps'),
    'mps_traced': partial(model_torch_traced, 'mps'),
    'm2_opt': get_model_m2_optimal,
}

if __name__ == '__main__':
    dset = 'ffhq256'
    show(run(CONFIGS['cuda'](dset)))
    show(run(CONFIGS['cpu'](dset)))
    show(run(CONFIGS['cpu_traced'](dset)))
    show(run(CONFIGS['mps'](dset)))
    show(run(CONFIGS['mps_traced'](dset)))
    show(run(CONFIGS['m2_opt'](dset)))
    print('Done')

    # M2 MacBook Air 13" (4E+4P+10GPU)
    # cpu         Lat:  46.85it/s, img: 0.46it/s, tot: 0.46it/s
    # cpu_traced  Lat: 170.92it/s, img: 0.47it/s, tot: 0.47it/s
    # mps         Lat:  20.82it/s, img: 2.99it/s, tot: 2.61it/s
    # mps_traced  Lat:  34.98it/s, img: 3.01it/s, tot: 2.77it/s
    # m2_opt      Lat:  62.44it/s, img: 2.98it/s, tot: 2.84it/s

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU)
    # cpu         Lat:  86.85it/s, img: 0.54it/s, tot: 0.54it/s
    # cpu_traced  Lat: 120.78it/s, img: 0.53it/s, tot: 0.53it/s
    # mps         Lat:  29.44it/s, img: 4.05it/s, tot: 3.56it/s
    # mps_traced  Lat:  40.38it/s, img: 4.19it/s, tot: 3.80it/s
    # m2_opt      Lat:  92.37it/s, img: 4.26it/s, tot: 4.07it/s (1.43x)
    # full_trace                                  tot: 4.24it/s