# Benchmark DiffAE performance on Apple M2 using different backends
# - Apache TVM (w/ optimizations)
# - coremltools (using NE)
# - Onnxruntime (CoreML, w/ NE?) - only C++?

import torch
import numpy as np
from model.unet_autoenc import AutoencReturn
from templates import LitModel
import templates_latent
from config import TrainMode
from PIL import Image
from tqdm import trange
import time

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

def get_samplers(conf, t_img=10, t_lat=10):
    conf_img = conf._make_diffusion_conf(conf.T)
    conf_img.use_timesteps = np.linspace(0, conf.T, max(2, t_img) + 1, dtype=np.int32).tolist()[:-1]
    sampl = conf_img.make_sampler()

    conf_lat = conf._make_latent_diffusion_conf(conf.T)
    conf_lat.use_timesteps = np.linspace(0, conf.T, max(2, t_lat) + 1, dtype=np.int32).tolist()[:-1]
    lat_sampl = conf_lat.make_sampler()

    return (sampl, lat_sampl)

def run(model, steps=10, B=1, verbose=True):
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
    latent_noise = torch.randn((B, 512)).to(dev_lat)
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

    # Run diffusion one step forward
    model_kwargs = {'x_start': None, 'cond': lats}
    
    # Initial value: spaial noise
    intermed = torch.randn((B, 3, conf.img_size, conf.img_size)).to(dev_img)

    for i in ran_fun(steps):
        t = torch.tensor([steps - i - 1] * B, device=dev_img, requires_grad=False)
        ret = sampl.ddim_sample(
            ema_model,
            intermed,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0.0,
        )
        intermed = ret['sample']

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
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    setattr(model, 'traced', False)
    
    return model.to(dev)

def model_torch_traced(dev, dset, B=1):
    model = model_torch(dev, dset)
    
    #model_forward = model.forward(x=x, t=self._scale_timesteps(t), **model_kwargs)
    x = torch.randn((B, 512)).to(dev)
    t = torch.tensor([9] * 1, device=dev, requires_grad=False) # unscaled

    # Latent net
    fwd_fun = lambda x, t: model.ema_model.latent_net(x, t).pred
    out_ref = fwd_fun(x, t)
    jit_lat = torch.jit.trace(fwd_fun, (x, t), check_trace=False)
    assert torch.allclose(out_ref.cpu(), jit_lat(x, t).cpu()), 'Lat-net outputs differ'
    
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
    #assert torch.allclose(out_ref.cpu(), jit_img(x, t, cond).cpu()), 'Img-net outputs differ'

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
