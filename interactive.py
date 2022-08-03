import os
import imgui
import torch
import argparse
import gdown
import numpy as np
from multiprocessing import Lock
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import reshape_grid, combo_box_vals
from typing import Dict, Tuple
from os import makedirs
from pathlib import Path
from glfw import KEY_LEFT_SHIFT

from bench import CONFIGS
from tracealbe import DiffAEModel

args = None
model_opts = ['bedroom128', 'horse128', 'ffhq256']

# Choose backend
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
mps = getattr(torch.backends, 'mps', None)
if mps and mps.is_available() and mps.is_built():
    device = 'mps'
    _orig = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda t: _orig(t.cpu()) # the whl is slightly buggy...

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_normal(shape=(1, 512), seed=None):
    seeds = sample_seeds(shape[0], base=seed)
    return seeds_to_samples(seeds, shape)

def seeds_to_samples(seeds, shape=(1, 512)):
    latents = np.zeros(shape, dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(shape[1:])
    
    return torch.tensor(latents)

class ModelViz(ToolbarViewer):    
    def __init__(self, name, batch_mode=False, hidden=False):
        self.batch_mode = batch_mode
        super().__init__(name, batch_mode=batch_mode, hidden=hidden)
    
    # Check for missing type annotations (break by-value comparisons)
    def check_dataclass(self, obj):
        for name in dir(obj):
            if name.startswith('__'):
                continue
            if name not in obj.__dataclass_fields__:
                raise RuntimeError(f'[ERR] Unannotated field: {name}')

    def setup_state(self):
        self.state = UIState()
        self.rend = RendererState()
        
        self.check_dataclass(self.state)
        self.check_dataclass(self.rend)
        self.state.pkl = args.model
        self.G_lock = Lock()

        # Optimized models:
        #  - fast latent sampling (jitted)
        #  - fast startup (img not jitted)
        default_backends = {
            'cpu': 'cpu',
            'mps': 'mps_opt',
            'cuda': 'cuda_opt',
        }

        self.state.backend = default_backends[device]
    
    @lru_cache()
    def _get_model(self, name, backend):
        return CONFIGS[backend](name)

    def init_model(self, name) -> DiffAEModel:
        model = self._get_model(name, self.state.backend)

        # Reset caches
        prev = self.rend.model
        if not prev or model.name != prev.name:
            self.rend.lat_cache = {}
            self.rend.img_cache = {}

        return model

    # Progress bar below images
    def draw_output_extra(self):
        self.rend.i = imgui.slider_int('', self.rend.i + 1, 1, self.rend.last_ui_state.T)[1] - 1

    def compute(self):
        # Copy for this frame
        s = deepcopy(self.state)

        # Perform computation
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.model = self.init_model(s.pkl)
            self.rend.i = 0
            res = self.rend.model.res
            dev = self.rend.model.dev_img
            self.rend.intermed = sample_normal((s.B, 3, res, res), s.seed).to(dev) # spaial noise
            if not self.rend.model.img_fused:
                self.rend.img_samp_params = self.rend.model.get_img_sampl_params(torch.tensor([s.T], device=dev))

        # Check if work is done
        if self.rend.i >= s.T - 1:
            return None

        model = self.rend.model
        cond = None
        if s.show_ds:
            # Use dataset image latents
            cond = self.rend.model.dset_lats[s.seed:s.seed+s.B].to(model.dev_img) # not normalized
        else:
            ################
            # Sample latents
            ################
            keys = [(s.seed + i, s.lat_T) for i in range(s.B)]
            missing = [k[0] for k in keys if k not in self.rend.lat_cache]
            if missing:
                latent_noise = seeds_to_samples(missing, (len(missing), 512)).to(model.dev_lat)
                lats = model.sample_lat_loop(torch.tensor([s.lat_T], device=model.dev_lat), latent_noise)
                
                # Update cache
                for seed, lat in zip(missing, lats):
                    self.rend.lat_cache[(seed, s.lat_T)] = lat
            
            cond = torch.stack([self.rend.lat_cache[k].to(model.dev_img) for k in keys], dim=0)

        # Run diffusion one step forward
        T = torch.tensor([s.T] * s.B, device=model.dev_img)
        t = T - self.rend.i - 1 # 0-based index, num_steps -> 0
        if model.img_fused:
            self.rend.intermed = model.sample_img_incr_fused(T, t, self.rend.intermed, cond)
        else:
            self.rend.intermed = model.sample_img_incr(t, self.rend.intermed, cond, *self.rend.img_samp_params)
        
        # Move on to next iteration
        self.rend.i += 1

        # Read from or write to cache
        finished = [False]*s.B
        for i, img in enumerate(self.rend.intermed):
            key = (s.show_ds, s.seed + i, s.T, s.lat_T)
            if key in self.rend.img_cache:
                self.rend.intermed[i] = torch.tensor(self.rend.img_cache[key], device=device)
                finished[i] = True
            elif self.rend.i >= s.T - 1:
                if not torch.any(torch.isnan(img)): # MPS bug: sometimes contains NaNs that darken image
                    self.rend.img_cache[key] = img.cpu().numpy()

        # Early exit
        if all(finished):
            self.rend.i = s.T - 1
        
        # Output updated grid
        grid = reshape_grid(0.5 * (self.rend.intermed + 1)) # => HWC
        return grid if grid.device.type == 'cuda' else grid.cpu().numpy()
    
    def draw_toolbar(self):
        jmp_large = 100 if self.v.keydown(KEY_LEFT_SHIFT) else 10

        s = self.state
        s.B = imgui.input_int('B', s.B)[1]
        s.seed = max(0, imgui.input_int('Seed', s.seed, s.B, 1)[1])
        s.show_ds = imgui.checkbox('Dataset latents', s.show_ds)[1]
        s.T = imgui.input_int('T_img', s.T, 1, jmp_large)[1]
        s.lat_T = imgui.input_int('T_lat', s.lat_T, 1, jmp_large)[1]
        s.pkl = combo_box_vals('Model', model_opts, s.pkl)[1]
        s.backend = combo_box_vals('Backend', list(CONFIGS.keys()), s.backend)[1]

# Volatile state: requires recomputation of results
@dataclass
class UIState:
    pkl: str = None
    T: int = 10
    lat_T: int = 100
    seed: int = 0
    B: int = 1
    show_ds: bool = False
    backend: str = None

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: DiffAEModel = None
    img_samp_params: Dict[str, torch.Tensor] = None
    intermed: torch.Tensor = None
    img_cache: Dict[Tuple[bool, int, int, int], torch.Tensor] = None
    lat_cache: Dict[Tuple[int, int], torch.Tensor] = None
    i: int = 0 # current computation progress

def init_torch():
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def download_models():
    files = [
        ('1EW94TP50IKqGPbPCL-Uld1TGnRtdH8aa', 'ffhq256_autoenc_latent/last.ckpt'),
        ('1y3cGbCIuMiGDyC6S-vt0SfoAxGP-GMPN', 'ffhq256_autoenc/latent.pkl')
    ]
    for id, suffix in files:
        pth = Path(__file__).parent / 'checkpoints' / suffix
        if not pth.is_file():
            makedirs(pth.parent, exist_ok=True)
            gdown.download(id=id, output=str(pth), quiet=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffAE visualizer')
    parser.add_argument('model', type=str, nargs='?', default='ffhq256', help='Model name [bedroom128  / horse128 / ffhq256]')
    args = parser.parse_args()
    assert args.model in model_opts, f'Unknown model {args.model}'

    download_models()
    init_torch()
    viewer = ModelViz('diffae_viewer', hidden=False)
    print('Done')
