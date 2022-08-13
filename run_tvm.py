import tvm # Built with BNNS support
from tvm import relay, autotvm
import relay_utils
import onnx

import os
import subprocess
import time
import numpy as np
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
from google.protobuf.json_format import MessageToDict
from bench import CONFIGS

#import numpy as np
#from pathlib import Path

lat_init = 'ckpts/ffhq256_lat_init.onnx'
lat_init_static = 'ckpts/ffhq256_lat_init_static10.onnx'
lat_step = 'ckpts/ffhq256_lat.onnx'
lat_step_fused = 'ckpts/ffhq256_lat_fused.onnx'
lat_norm = 'ckpts/ffhq256_lat_norm.onnx'
img_init = 'ckpts/ffhq256_img_init.onnx'
img_step = 'ckpts/ffhq256_img.onnx'
img_step_fused = 'ckpts/ffhq256_img_fused.onnx'

def compile_tvmc(mod_name):
    mod = onnx.load(mod_name)

    in_shapes = []
    out_shapes = []
    for nodes, shp, type in [(mod.graph.input, in_shapes, 'input'), (mod.graph.output, out_shapes, 'output')]:
        for _node in nodes:
            m_dict = MessageToDict(_node)
            dim_info = m_dict.get('type').get('tensorType').get('shape').get('dim')
            var_shape = [d.get('dimValue', '?') for d in dim_info]  # [4,3,384,640]
            shape_str = ','.join(var_shape)
            shp.append(f'{_node.name}:[{shape_str}]')
            if '?' in var_shape:
                print(f'WARN - {_node.name} ({type}): non-static shape [{shape_str}]')
    
    shapes_str = ','.join(in_shapes)
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

def compile_autotvm(mod_name, ops=()):
    import tvm.relay.testing
    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    import tvm.contrib.graph_executor as runtime

    batch_size = 2
    input_shape = (batch_size, 1)
    output_shape = (batch_size, 3, 256, 256)

    print('Loading ONNX model')
    onnx_mod = onnx.load(mod_name)

    print('Creating relay model')
    print('TODO: cache this!')
    mod, params = relay.frontend.from_onnx(onnx_mod)
    del onnx_mod

    # BNNS
    #mod = relay.op.contrib.bnns.partition_for_bnns(mod)

    #### DEVICE CONFIG ####
    target = tvm.target.Target('metal')
    #target = tvm.target.Target("metal -mcpu=apple-latest -mtriple=arm64-apple-macos")
    #target = tvm.target.Target("llvm -mcpu=apple-latest -mtriple=arm64-apple-macos") # target_host?

    # Get tasks
    # nn.dense, nn.conv2d, nn.bias_add
    print('Extracting tasks...')
    ops = ['nn.conv2d', 'nn.dense', 'nn.bias_add'] + list(ops)
    tasks = autotvm.task.extract_from_program(
        mod['main'], target=target, params=params, ops=[relay.op.get(n) for n in ops]
    )
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
    #devs = subprocess.check_output('python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190'.split(' ')).decode('utf-8')
    #print(devs)

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
        'n_trial': 150, #1500,
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

    # run tuning tasks
    print('Tuning...')
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print('Compile...')
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = 'net.so'
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = 'net.tar'
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print('Upload...')
        remote = autotvm.measure.request_remote(device_key, '127.0.0.1', 9190, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib['default'](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)

        # evaluate
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

if __name__ == '__main__':
    #_ = CONFIGS['static_lat_init']('ffhq256', export=True)
    #_ = CONFIGS['cpu_traced']('ffhq256', export=True)
    #compile_tvmc(lat_init_static)
    #compile_tvmc(lat_init)
    #compile_tvmc(lat_step_fused) # Cannot allocate memory symbolic tensor shape [?]
    compile_autotvm(img_step_fused)

    print('Done')
