import tvm # built with BNNS support
from tvm import relay, autotvm 
import onnx

import os
import subprocess
import hashlib
import warnings
import time
import numpy as np
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
from google.protobuf.json_format import MessageToDict
from bench import CONFIGS
from pathlib import Path

def get_shapes_onnx(mod: onnx.ModelProto):
    in_shapes = {}
    out_shapes = {}
    for nodes, shp in [(mod.graph.input, in_shapes), (mod.graph.output, out_shapes)]:
        for _node in nodes:
            m_dict = MessageToDict(_node)
            dim_info = m_dict.get('type').get('tensorType').get('shape').get('dim')
            var_shape = [d.get('dimValue', '?') for d in dim_info]  # [4,3,384,640]
            shp[_node.name] = var_shape

    return in_shapes, out_shapes

def get_shapes_relay(mod: tvm.IRModule):
    shape_in = {}
    shape_out = None
    try:
        assert len(mod.functions) == 1, 'Multiple functions in module'
        inputs = next(mod.functions.values()).params
        shape_in = { v.name_hint : [int(d) for d in v.checked_type.shape] for v in inputs }
    except:
        pass
    
    return shape_in, shape_out

def compile_tvmc(mod_name):
    in_shapes, out_shapes = get_shapes_onnx(onnx.load(mod_name))
    for shapes, vtype in [(in_shapes, 'input'), (out_shapes, 'output')]:
        for k, v in shapes.items():
            if '?' in v:
                print(f'WARN - {k} ({vtype}): non-static shape [{v}]')

    shapes_str = ','.join([f'{n}:[{",".join(s)}]' for n,s in in_shapes.items()])
    cmd = f'tvmc -v compile --target "llvm" --input-shapes "{shapes_str}" --output {mod_name[:-5]}-tvm.tar {mod_name}'
    print(cmd)
    
    if subprocess.call(cmd, shell=True):
        print('ERROR!')
    else:
        print('Success!')


def tune_tasks(
    tasks,
    measure_option,
    tuner='xgb',
    n_trial=1000,
    early_stopping=None,
    log_filename='tuning.log',
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + '.tmp'
    if os.path.exists(tmp_log_file):
        if not use_transfer_learning:
            raise RuntimeError('Logfile exists: ' + tmp_log_file)
        print('Resuming from existing log')

    for i, tsk in enumerate(reversed(tasks)):
        prefix = '[Task %2d/%2d] ' % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'xgb_itervar':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='itervar')
        elif tuner == 'xgb_curve':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='curve')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError('Invalid tuner: ' + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

# ONNX to relay translation is slow
#  => cache results
def load_relay_cached(model_path: str, batch_dim = 1):
    cache = Path(__file__).parent / 'ckpts' / 'cache'
    
    print('MD5: ', end='')
    hash_object = hashlib.md5()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<16), b''):
            hash_object.update(chunk)
    md5 = hash_object.hexdigest()
    print(md5)
    
    model_name = Path(model_path).stem + f'_{batch_dim}'
    cache_dir = cache / md5
    os.makedirs(cache_dir, exist_ok=True)
    
    graph = cache_dir / (model_name + '_relay.json')
    param = cache_dir / (model_name + '_relay.params')

    # Cache
    if not graph.is_file() or not param.is_file():
        model = onnx.load(model_path)
        model.__class__.__repr__ = lambda _ : 'dummy'
        in_shapes, _ = get_shapes_onnx(model)
        for k, v in in_shapes.items():
            if len(v) > 1:
                if v[0] not in [str(batch_dim), '?']:
                    raise RuntimeError(f'Incompatible static batch dim for {k}: {v}')
                v[0] = batch_dim
            in_shapes[k] = [int(s) for s in v]

        # freeze_params converts dynamic shapes to static ones
        mod, params = relay.frontend.from_onnx(model, shape=in_shapes, freeze_params=True)
        graph.write_text(tvm.ir.save_json(mod))
        param.write_bytes(relay.save_param_dict(params))

    # Load
    try:
        mod = tvm.ir.load_json(graph.read_text())
        mod.__class__.__orig_repr__ = mod.__class__.__repr__ 
        mod.__class__.__repr__ = lambda _: '__repr__ suppressed' # patch for laggy pydevd
        params = relay.load_param_dict(param.read_bytes())
    except RuntimeError as e:
        print(e)
    
    print('Relay model: applying optimizations')
    mod = tvm.relay.transform.FastMath()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
        tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    mod = BindPass(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    # TEST
    mod = tvm.relay.transform.DynamicToStatic()(mod) # in onnx loader
    #mod = tvm.relay.transform.ToMixedPrecision()(mod)

    return mod, params

def compile_autotvm(mod_name, ops=()):
    import tvm.relay.testing
    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    import tvm.contrib.graph_executor as runtime

    # TVM currently does not support dynamic input shapes
    # Relay Next will change this (github.com/tlc-pack/relax/wiki/Relax-Architecture-Overview)
    batch_size = 1
    input_shape = (batch_size, 1)
    output_shape = (batch_size, 3, 256, 256)

    print('Loading relay model')
    mod, params = load_relay_cached(mod_name, batch_size)

    # BNNS
    #mod = relay.op.contrib.bnns.partition_for_bnns(mod)

    #### DEVICE CONFIG ####
    target = tvm.target.Target('metal', host='llvm -mcpu=apple-latest -mtriple=arm64-apple-macos')
    #target = tvm.target.Target("llvm -mcpu=apple-latest -mtriple=arm64-apple-macos") # target_host?

    # Test: run model (1)
    shapes, _ = get_shapes_relay(mod)
    data = { k: np.random.randn(*v).astype(np.float32) for k,v in shapes.items() }

    # with tvm.transform.PassContext(opt_level=0):
    #     # Graph compiler erros due to dynamic shapes...?
    #     # executor = relay.build_module.create_executor(
    #     #     "graph", mod, tvm.cpu(0), 'llvm', params
    #     # ).evaluate()
        
    #     # VM supports dynamic shapes
    #     executor = relay.build_module.create_executor(
    #         'vm', mod, tvm.cpu(0), 'llvm', params
    #     ).evaluate()
        
    #     tvm_output = executor(data)
    #     print(tvm_output)

    # compiler = relay.vm.VMCompiler()
    # compiler.set_params(params)
    # #compiler.lower(mod, target=target)
    # opt_mod, _ = compiler.optimize(mod, target=target)
    # print(opt_mod)


    # vmc = relay.vm.VMCompiler()
    # vm = vmc.lower(mod, "llvm")
    # vm = tvm.relay.vm.compile(mod, target='llvm')

    #vm.init(ctx)
    #vm.load_params(params)
    # shapes, _ = get_shapes_relay(mod)
    # data = { k: np.random.randn(*v).astype(np.float32) for k,v in shapes.items() }
    # out = vm.run(data)
    # print(out)

    # Test: run model 
    #print('Exporting test lib')
    #lib = relay.build(mod, target=target, params=params) # targets graph executor
    #lib.export_library(f'test_model_{target.kind.name}.dylib')

    # Get tasks
    # nn.dense, nn.conv2d, nn.bias_add
    print('Extracting tasks...')
    ops = ['nn.conv2d', 'nn.dense', 'nn.bias_add'] + list(ops)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        tasks = autotvm.task.extract_from_program(
            mod['main'], target=target, params=params, ops=[relay.op.get(n) for n in ops])
    
    assert tasks, 'No tasks found!'
    print('Found', len(tasks), 'tasks')

    # Start runners
    print('Starting rpc tracker and server')
    ps_listing = subprocess.check_output('ps').decode('utf-8')
    
    if not '--file tvm.exec.rpc_tracker' in ps_listing:
        subprocess.Popen(['python', '-m', 'tvm.exec.rpc_tracker', '--host', '0.0.0.0', '--port', '9190'])
        time.sleep(2)
    if not '--file tvm.exec.rpc_server' in ps_listing:
        subprocess.Popen(['python', '-m', 'tvm.exec.rpc_server', '--tracker', '127.0.0.1:9190', '--port', '9090', '--key', 'm1', '--no-fork'])
        time.sleep(2)

    # Query devices
    devs = subprocess.check_output(
        'python -m tvm.exec.query_rpc_tracker --host 0.0.0.0 --port 9190'.split(' ')).decode('utf-8')
    print(devs)

    #### TUNING OPTION ####
    network = mod_name
    log_file = '%s.log' % network
    dtype = 'float32'

    # Also replace this with the device key in your tracker
    device_key = 'm1'

    # Set this to True if you use android phone
    use_android = False

    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb', # 'xgb', 'xgb_knob', 'xgb_itervar', 'xgb_curve', 'ga', 'random', 'gridsearch'
        'n_trial': 15, #1500,
        'early_stopping': 800,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func='ndk' if use_android else 'default'),
            # CUDA: runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)
            runner=autotvm.RPCRunner(
                device_key,
                host='127.0.0.1',
                port=9190,
                number=5,
                timeout=10,
            ),
        ),
    }

    # Run tuning tasks
    print('Tuning...')
    tune_tasks(tasks, **tuning_option)

    # Compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print('Compile...')
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # Export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = 'net.so'
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = 'net.tar'
            lib.export_library(tmp.relpath(filename))

        # Upload module to device
        print('Upload...')
        remote = autotvm.measure.request_remote(device_key, '127.0.0.1', 9190, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # Upload parameters to device
        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib['default'](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)

        # Evaluate
        print('Evaluate inference time cost...')
        print(module.benchmark(dev, number=1, repeat=10))

    


# class SilentProto(onnx.ModelProto):
#     def __str__(self):
#         return 'DISABLED'

# Latent net: use BNNS for fast matrix multiply (CPU only)
#mod_onnx = onnx.load(lat_step_fused)
#mod, params = relay.frontend.from_onnx(onnx.load(lat_step), freeze_params=True)
#mod = relay.op.contrib.bnns.partition_for_bnns(mod)
#lib = relay.build(mod, target='llvm', params=params)
#lib.export_library('model_with_bnns.dylib')

#relay_utils.dump_pt()

#mod, params, shape_dict = relay_utils.load_pt_model(img_step.replace('onnx', 'pt'))

#compile_tvmc(lat_norm) # OK

lat_init = 'ckpts/ffhq256_lat_init.onnx'
lat_init_static = 'ckpts/ffhq256_lat_init_static10.onnx'
lat_step = 'ckpts/ffhq256_lat.onnx'
lat_step_fused = 'ckpts/ffhq256_lat_fused.onnx'
lat_norm = 'ckpts/ffhq256_lat_norm.onnx'
img_init = 'ckpts/ffhq256_img_init.onnx'
img_step = 'ckpts/ffhq256_img.onnx'
img_step_fused = 'ckpts/ffhq256_img_fused.onnx'

if __name__ == '__main__':
    #_ = CONFIGS['static_lat_init']('ffhq256', export=True)
    #_ = CONFIGS['cpu_traced']('ffhq256', export=True)
    #compile_tvmc(lat_init_static)
    #compile_tvmc(lat_step_fused) # Cannot allocate memory symbolic tensor shape [?]
    #compile_autotvm(img_step_fused)
    #compile_autotvm(lat_init_static) # OK!
    #compile_autotvm(lat_init) # Cannot allocate memory symbolic tensor shape [?]
    #compile_autotvm(lat_step_fused) # dynamic linspace => error
    
    # Fully statc shapes
    compile_tvmc(lat_step)
    compile_tvmc(img_step)
    compile_autotvm(lat_step)
    compile_autotvm(img_step)

    # TODO: fused models that take pre-constructed linspace in

    print('Done')
