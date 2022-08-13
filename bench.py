# Benchmark DiffAE performance on Apple M2 using different backends
# - Apache TVM (w/ optimizations)
# - coremltools (using NE)
# - Onnxruntime (CoreML, w/ NE?) - only C++?
# - Wonnx (WebGPU / Metal / Vulkan)

import torch
import numpy as np
from numpy.random import RandomState
from PIL import Image
from tqdm import trange
import random
import time
import re
from os import makedirs
from pathlib import Path

from tracealbe import DiffAEModel
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

def run(model: DiffAEModel, steps=200, B=1, verbose=True, seed=None):
    random.seed(time.time())
    seed = random.randint(0, 1<<32-1) if seed is None else seed
    steps_lat = 200
    dev_lat = model.dev_lat
    dev_img = model.dev_img
    
    # Warmup
    _x1 = torch.randn((B, 512)).to(dev_lat)
    _x2 = torch.randn((B, 3, model.res, model.res)).to(dev_img)
    _t = torch.tensor([3] * B)
    print('Warmup', end='')
    for _ in range(3):
        _ = model.forward(_t.to(dev_lat), _t.to(dev_img), _x1, _x2)
        print('.', end='')
    print('done')

    t0 = time.time()
    latent_noise = torch.tensor(RandomState(seed).randn(B, 512), dtype=torch.float32).to(dev_lat)
    T = torch.tensor([steps_lat], dtype=torch.int64, device=dev_lat)
    lats = model.sample_lat_loop(T, latent_noise).to(dev_img)
    t1 = time.time()
    
    # Initial value: spaial noise
    intermed = torch.tensor(RandomState(seed).randn(B, 3, model.res, model.res), dtype=torch.float32).to(dev_img)
    intermed = model.sample_img_loop(T.to(dev_img), intermed, lats)
    t2 = time.time()
    
    it_lat = (t1 - t0) / steps_lat
    it_img = (t2 - t1) / steps
    it_tot = (t2 - t0) / steps

    # Summary
    dev_desc = []
    for d, mod in zip([dev_lat, dev_img], [model.lat_net.mode, model.img_net.mode]):
        dev_desc.append(f'[{mod}]{d.type.upper()}')
    summary = f'{"+".join(dev_desc)} - Lat: {1/it_lat:.2f}it/s, img: {1/it_img:.2f}it/s, tot: {1/it_tot:.2f}it/s'
    
    if verbose:
        print(summary)

    return intermed, summary

# Fusing only makes sense for traced models
def _get_model(dev_lat, dev_img, dset, fp16=False, lat_fused=False, img_fused=False):
    model = DiffAEModel(dset, dev_lat, dev_img, fp16, lat_fused=lat_fused, img_fused=img_fused)
    setattr(model.lat_net, 'mode', 'INT')
    setattr(model.img_net, 'mode', 'INT')
    return model

def model_torch(dev_lat, dev_img, dset, fp16=False):
    return _get_model(dev_lat, dev_img, dset, fp16, False, False)

def test_static(
    dev_lat,
    dev_img,
    dset,
    fp16=False,
    B=1,
    trace_lat=True,
    trace_img=True,
    fuse_lat=False,
    fuse_img=False,
    export=False
) -> DiffAEModel:
    makedirs(Path(__file__).parent / 'ckpts', exist_ok=True)
    model = _get_model(dev_lat, dev_img, dset, fp16, fuse_lat, fuse_img)
    suff = '_static10'

    T = torch.tensor([10] * B, dtype=torch.int64).view(-1, 1).to(dev_lat)
    t = T - 1
    
    # Latent net init
    fwd_fun = lambda T : model.lat_sampl.forward_static_10(T)
    jit_lat_init = torch.jit.trace(fwd_fun, (T), check_trace=False)
    if export:
        jit_lat_init.save(f'ckpts/{dset}_lat_init{suff}.pt')
        torch.onnx.export(
            jit_lat_init,                 # model being run
            (T),                          # model input (or a tuple for multiple inputs)
            f'ckpts/{dset}_lat_init{suff}.onnx',      # where to save the model (can be a file or file-like object)
            input_names = ['T'],          # the model's input names
            output_names = [              # the model's output names
                'timestep_map',
                'alphas_cumprod',
                'alphas_cumprod_prev',
                'sqrt_recip_alphas_cumprod',
                'sqrt_recipm1_alphas_cumprod',
            ]
        )

        print('Exported static lat_init')


# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
def model_torch_traced(
    dev_lat,
    dev_img,
    dset,
    fp16=False,
    B=1,
    trace_lat=True,
    trace_img=True,
    fuse_lat=False,
    fuse_img=False,
    export=False
) -> DiffAEModel:
    makedirs(Path(__file__).parent / 'ckpts', exist_ok=True)
    model = _get_model(dev_lat, dev_img, dset, fp16, fuse_lat, fuse_img)
    check_trace = 'mps' not in [dev_img, dev_lat]
    suff = '_fp16' if fp16 else ''
    
    if is_set('USE_MKLDNN') and is_set('BUILD_ONEDNN_GRAPH'):
        torch.jit.enable_onednn_fusion(True)

    T = torch.tensor([10] * B, dtype=torch.int64).view(-1, 1).to(dev_lat)
    t = T - 1

    # torch.jit.trace(lambda) => TraceFunction
    # torch.jit.trace(module.fun) => trace doesn't support compiling individual module's functions
    # torch.jit.trace_module(module, {'forward': (i1, i2)}) => Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults
    # torch.jit.trace(module) = torch.jit.trace(module.forward)?
    # torch.jit.trace_module(module) => TraceModule?
    # 'Iterating over a tensor might cause the trace to be incorrect.'

    # example_outputs no longer exists in torch 1.11+

    if trace_lat:

        # Latent net init
        if not model.lat_fused:
            fwd_fun = lambda T : model.lat_sampl.forward(T)
            jit_lat_init = torch.jit.trace(fwd_fun, (T), check_trace=check_trace)
            if export:
                jit_lat_init.save(f'ckpts/{dset}_lat_init{suff}.pt')
                torch.onnx.export(
                    jit_lat_init,                 # model being run
                    (T),                          # model input (or a tuple for multiple inputs)
                    f'ckpts/{dset}_lat_init{suff}.onnx',      # where to save the model (can be a file or file-like object)
                    input_names = ['T'],          # the model's input names
                    output_names = [              # the model's output names
                        'timestep_map',
                        'alphas_cumprod',
                        'alphas_cumprod_prev',
                        'sqrt_recip_alphas_cumprod',
                        'sqrt_recipm1_alphas_cumprod',
                    ]
                )

            # Replace with jitted
            setattr(model.lat_sampl, 'orig_fwd', model.lat_sampl.forward)
            model.lat_sampl.forward = jit_lat_init

        # Latent net
        x0 = torch.randn(B, 512).to(dev_lat)
        jit_lat = None
        input_names = []
        model_name = ''
        ex_inputs = []
        
        if model.lat_fused:
            fwd_fun = lambda T, t, x : model.sample_lat_incr_fused(T, t, x)
            ex_inputs = (T, t, x0)
            input_names = ['T', 't', 'x']
            jit_lat = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
            model_name = f'{dset}_lat_fused'
            setattr(model, 'orig_sample_lat_incr_fused', model.sample_lat_incr_fused)
            model.sample_lat_incr_fused = jit_lat
        else:
            fwd_fun = lambda t, x, *vs : model.lat_sampl.sample_incr(t, x, model.lat_net, *vs)
            ex_inputs = (t, x0, *model.lat_sampl.forward(T))
            input_names = ['t', 'x', 'timestep_map', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod']
            jit_lat = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
            model_name = f'{dset}_lat'
            setattr(model.lat_sampl, 'orig_sample_incr', model.lat_sampl.sample_incr)
            model.lat_sampl.sample_incr = lambda t, x, _, *params : jit_lat(t, x, *params)

        if export:
            jit_lat.save(f'ckpts/{model_name}{suff}.pt')
            torch.onnx.export(
                jit_lat,                  # model being run
                ex_inputs,                # model input (or a tuple for multiple inputs)
                f'ckpts/{model_name}{suff}.onnx',     # where to save the model (can be a file or file-like object)
                input_names = input_names,
                output_names = ['lats'],
                #do_constant_folding=False
            )

        # Latent normalization function
        norm_fun = lambda lats : model.lat_denorm(lats)
        jit_norm = torch.jit.trace(norm_fun, (torch.randn(B, 512)))
        if export:
            jit_norm.save(f'ckpts/{dset}_lat_norm{suff}.pt')
            torch.onnx.export(jit_norm, (torch.randn(B, 512)),
                f'ckpts/{dset}_lat_norm.onnx', input_names=['lats_in'], output_names=['lats'])

        setattr(model.lat_net, 'mode', 'JIT')

    if trace_img:

        # Image net init
        if not model.img_fused:
            T = T.to(dev_img)
            fwd_fun = lambda T : model.img_sampl.forward(T)
            jit_img_init = torch.jit.trace(fwd_fun, (T), check_trace=check_trace)
            if export:
                jit_img_init.save(f'ckpts/{dset}_img_init{suff}.pt')
                torch.onnx.export(
                    jit_img_init,                 # model being run
                    (T),                          # model input (or a tuple for multiple inputs)
                    f'ckpts/{dset}_img_init{suff}.onnx',      # where to save the model (can be a file or file-like object)
                    input_names = ['T'],          # the model's input names
                    output_names = [              # the model's output names
                        'timestep_map',
                        'alphas_cumprod',
                        'alphas_cumprod_prev',
                        'sqrt_recip_alphas_cumprod',
                        'sqrt_recipm1_alphas_cumprod',
                    ]
                )

            # Replace with jitted
            setattr(model.img_sampl, 'orig_fwd', model.img_sampl.forward)
            model.img_sampl.forward = jit_img_init

        # Image net
        jit_img = None
        input_names = []
        model_name = ''
        ex_inputs = []

        lats = torch.randn(B, 512).to(dev_img)
        T = T.to(dev_img)
        t = t.to(dev_img)
        x0 = torch.randn(B, 3, model.res, model.res).to(dev_img)
        
        if model.img_fused:
            fwd_fun = lambda T, t, x, lats : model.sample_img_incr_fused(T, t, x, lats)
            ex_inputs = (T, t, x0, lats)
            input_names = ['T', 't', 'x', 'lats']
            jit_img = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
            model_name = f'{dset}_img_fused'
            setattr(model, 'orig_sample_img_incr_fused', model.sample_img_incr_fused)
            model.sample_img_incr_fused = jit_img
        else:
            fwd_fun = lambda t, x, lats, *vs : model.img_sampl.sample_incr(t, x, partial(model.img_net, cond=lats), *vs)
            ex_inputs = (t, x0, lats, *model.img_sampl.forward(T))
            input_names = ['t', 'x', 'lats', 'timestep_map', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod']
            jit_img = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
            model_name = f'{dset}_img'
            setattr(model.img_sampl, 'orig_sample_incr', model.img_sampl.sample_incr)
            model.img_sampl.sample_incr = lambda t, x, mod, *params : jit_img(t, x, mod.keywords['cond'], *params)

        if export:
            jit_img.save(f'ckpts/{model_name}{suff}.pt')
            torch.onnx.export(
                jit_img,                     # model being run
                ex_inputs,                   # model input (or a tuple for multiple inputs)
                f'ckpts/{model_name}{suff}.onnx',        # where to save the model (can be a file or file-like object)
                input_names = input_names,
                output_names = ['output'],
                #do_constant_folding=False,
                #keep_initializers_as_inputs=False,
                #export_modules_as_functions=False,
                #verbose=False,
                #operator_export_type=torch.onnx.OperatorExportTypes.ONNX, # ONNX_ATEN, ONNX_ATEN_FALLBACK
            )

        setattr(model.img_net, 'mode', 'JIT')
    
    return model

def load_pt_model(dev_lat, dev_img, dset, fp16=False):
    model = DiffAEModel(dset, dev_lat, dev_img)
    suff = '_fp16' if fp16 else ''

    model.lat_sampl.forward = torch.jit.load(f'{dset}_lat_init{suff}.pt').to(dev_lat)
    model.img_sampl.forward = torch.jit.load(f'{dset}_img_init{suff}.pt').to(dev_img)
    model.lat_sampl.sample_incr = torch.jit.load(f'{dset}_lat{suff}.pt').to(dev_lat)
    model.img_sampl.sample_incr = torch.jit.load(f'{dset}_img{suff}.pt').to(dev_img)

    setattr(model.lat_net, 'mode', 'PTH')
    setattr(model.img_net, 'mode', 'PTH')
    return model


from functools import partial
CONFIGS = {
    'cuda': partial(model_torch, 'cuda', 'cuda'),
    'cuda_fp16': partial(model_torch, 'cuda', 'cuda', fp16=True),
    'cuda_traced': partial(model_torch_traced, 'cuda', 'cuda'),
    'cuda_traced_fp16': partial(model_torch_traced, 'cuda', 'cuda', fp16=True),
    'cuda_fused': partial(model_torch_traced, 'cuda', 'cuda', fuse_lat=True, fuse_img=True),
    'cuda_fused_fp16': partial(model_torch_traced, 'cuda', 'cuda', fuse_lat=True, fuse_img=True, fp16=True),
    'cuda_pt': partial(load_pt_model, 'cuda', 'cuda'),
    'cuda_pt_fp16': partial(load_pt_model, 'cuda', 'cuda', fp16=True),
    'cuda_opt': partial(model_torch_traced, 'cuda', 'cuda', trace_img=False),
    'cuda_opt_fp16': partial(model_torch_traced, 'cuda', 'cuda', trace_img=False, fp16=True),
    'cpu': partial(model_torch, 'cpu', 'cpu'),
    'cpu_traced': partial(model_torch_traced, 'cpu', 'cpu'),
    'cpu_fused': partial(model_torch_traced, 'cpu', 'cpu', fuse_lat=True, fuse_img=True),
    'cpu_pt': partial(load_pt_model, 'cpu', 'cpu'),
    'mps': partial(model_torch, 'mps', 'mps'),
    'mps_traced': partial(model_torch_traced, 'mps', 'mps'),
    'mps_fused': partial(model_torch_traced, 'mps', 'mps', fuse_lat=True, fuse_img=True),
    'mps_pt': partial(load_pt_model, 'mps', 'mps'),
    'mps_opt': partial(model_torch_traced, 'mps', 'mps', trace_img=False),
    'm2_opt': partial(model_torch_traced, 'cpu', 'mps'),
    'static_lat_init': partial(test_static, 'cpu', 'cpu')
}

if __name__ == '__main__':
    dset = 'ffhq256'
    mps = getattr(torch.backends, 'mps', None)
    seed = random.randint(0, 1<<32-1)
    
    configs = []
    if torch.cuda.is_available():
        configs.append('cuda')
        configs.append('cuda_opt')
        configs.append('cuda_traced')
        configs.append('cuda_fused')
        configs.append('cuda_pt')
        configs.append('cuda_fp16')
        configs.append('cuda_opt_fp16')
        configs.append('cuda_traced_fp16')
        configs.append('cuda_fused_fp16')
        configs.append('cuda_pt_fp16')
    
    if mps and mps.is_available() and mps.is_built():
        configs.append('mps')
        configs.append('mps_opt')
        configs.append('mps_traced')
        configs.append('mps_fused')
        configs.append('mps_pt')
        configs.append('m2_opt')
    
    configs.append('cpu')
    configs.append('cpu_traced')
    configs.append('cpu_fused')
    configs.append('cpu_pt')

    summaries = []
    for cfg in configs:
        print('Running', cfg)
        img, summ = run(CONFIGS[cfg](dset), seed=seed)
        summaries.append(summ)
        show(img)

    print('Results:')
    for conf, summ in zip(configs, summaries):
        print(f'{conf}: {summ}')

    print('Done')

    # Original code (Numpy + Torch)

    # M2 MacBook Air 13" (4E+4P+10GPU)
    # cpu               Lat:   46.85it/s, img:  0.46it/s, tot:  0.46it/s
    # cpu_traced        Lat:  170.92it/s, img:  0.47it/s, tot:  0.47it/s
    # mps               Lat:   20.82it/s, img:  2.99it/s, tot:  2.61it/s
    # mps_traced        Lat:   34.98it/s, img:  3.01it/s, tot:  2.77it/s
    # m2_opt            Lat:   62.44it/s, img:  2.98it/s, tot:  2.84it/s

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU)
    # cpu               Lat:   86.85it/s, img:  0.54it/s, tot:  0.54it/s
    # cpu_traced        Lat:  120.78it/s, img:  0.53it/s, tot:  0.53it/s
    # mps               Lat:   29.44it/s, img:  4.05it/s, tot:  3.56it/s
    # mps_traced        Lat:   40.38it/s, img:  4.19it/s, tot:  3.80it/s
    # m2_opt            Lat:   92.37it/s, img:  4.26it/s, tot:  4.07it/s
    # full_trace (static loops)                           tot:  4.24it/s

    # Torch-only rewrite (1.8.2022)

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU) - no debugger
    # cpu               Lat:   90.62it/s, img:  0.50it/s, tot:  0.50it/s
    # cpu_traced        Lat:  118.95it/s, img:  0.52it/s, tot:  0.52it/s
    # mps               Lat:   71.70it/s, img:  4.41it/s, tot:  4.15it/s
    # mps_traced        Lat:  107.92it/s, img:  4.44it/s, tot:  4.26it/s
    # m2_opt            Lat:   88.16it/s, img:  4.35it/s, tot:  4.14it/s
    
    # M1 Pro MacBook Pro 14" (2E+6P+14GPU) - with debugger
    # cpu               Lat:   51.82it/s, img:  0.50it/s, tot:  0.49it/s
    # cpu_traced        Lat:  121.03it/s, img:  0.50it/s, tot:  0.50it/s
    # mps               Lat:   51.25it/s, img:  4.28it/s, tot:  3.95it/s
    # mps_traced        Lat:  101.61it/s, img:  4.35it/s, tot:  4.17it/s
    # m2_opt            Lat:   88.10it/s, img:  4.41it/s, tot:  4.20it/s

    # Increates sample counts to 200+200

    # i5-12600k + RTX 2080 - no debugger
    # cuda              Lat:  222.28it/s, img:  6.96it/s, tot:  6.75it/s
    # cuda_traced       Lat:  876.19it/s, img:  7.10it/s, tot:  7.05it/s
    # cuda_fused        Lat:  814.24it/s, img:  7.15it/s, tot:  7.09it/s
    # cuda_pt           Lat:  214.04it/s, img:  6.85it/s, tot:  6.63it/s
    # cpu               Lat:  119.99it/s, img:  0.63it/s, tot:  0.63it/s
    # cpu_traced        Lat:  225.53it/s, img:  0.64it/s, tot:  0.64it/s
    # cpu_fused         Lat:  219.16it/s, img:  0.65it/s, tot:  0.64it/s
    # cpu_pt            Lat:  129.14it/s, img:  0.66it/s, tot:  0.66it/s

    # Device transfer optimizations, added fp16 (3.8.2022)

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU) - no debugger
    # cpu               Lat:   92.76it/s, img:  0.52it/s, tot:  0.51it/s
    # cpu_traced        Lat:  125.21it/s, img:  0.54it/s, tot:  0.53it/s
    # cpu_fused         Lat:  123.99it/s, img:  0.54it/s, tot:  0.54it/s
    # cpu_pt            Lat:   92.34it/s, img:  0.54it/s, tot:  0.54it/s
    # mps               Lat:  114.75it/s, img:  4.44it/s, tot:  4.27it/s
    # mps_traced        Lat:  182.43it/s, img:  4.51it/s, tot:  4.40it/s
    # mps_fused         Lat:  143.47it/s, img:  4.37it/s, tot:  4.24it/s
    # mps_pt            Lat:  114.61it/s, img:  4.59it/s, tot:  4.42it/s
    # mps_opt           Lat:  189.42it/s, img:  4.45it/s, tot:  4.35it/s
    # m2_opt            Lat:  123.58it/s, img:  4.58it/s, tot:  4.42it/s

    # i5-12600k + RTX 2080 - no debugger
    # cuda              Lat:  212.83it/s, img:  6.86it/s, tot:  6.64it/s
    # cuda_opt          Lat:  663.59it/s, img:  6.64it/s, tot:  6.58it/s
    # cuda_traced       Lat:  603.12it/s, img:  7.20it/s, tot:  7.11it/s
    # cuda_fused        Lat:  588.14it/s, img:  7.14it/s, tot:  7.05it/s
    # cuda_pt           Lat:  194.82it/s, img:  6.76it/s, tot:  6.53it/s
    # cuda_fp16         Lat:  200.60it/s, img: 13.61it/s, tot: 12.74it/s
    # cuda_pt_fp16      Lat:  206.73it/s, img:  6.62it/s, tot:  6.41it/s
    # cuda_opt_fp16     Lat: 1046.24it/s, img: 12.62it/s, tot: 12.47it/s
    # cuda_traced_fp16  Lat: 1009.67it/s, img: 15.13it/s, tot: 14.91it/s
    # cuda_fused_fp16   Lat:  882.46it/s, img: 15.27it/s, tot: 15.01it/s
    # cpu               Lat:  124.31it/s, img:  0.61it/s, tot:  0.61it/s
    # cpu_traced        Lat:  219.77it/s, img:  0.62it/s, tot:  0.62it/s
    # cpu_fused         Lat:  220.07it/s, img:  0.62it/s, tot:  0.61it/s
    # cpu_pt            Lat:  124.18it/s, img:  0.61it/s, tot:  0.61it/s
    
