# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import inspect
import os
import subprocess
import sys
import timeit
import argparse
import copy
import re

import libconf
import yaml

from common import *

# Output file names.
out_prefix = "timeloop-mapper."
log_file_name = out_prefix + "log"
stats_file_name = out_prefix + "stats.txt"
xml_file_name = out_prefix + "map+stats.xml"
map_txt_file_name = out_prefix + "map.txt"
map_cfg_file_name = out_prefix + "map.cfg"
map_cpp_file_name = out_prefix + "map.cpp"
output_file_names = [log_file_name,
                     stats_file_name,
                     xml_file_name,
                     map_txt_file_name,
                     map_cfg_file_name,
                     map_cpp_file_name]

# dimension conversion that maps a WU problem to FW problem
wu2fw = {'P': 'R',
         'Q': 'S',
         'R': 'P',
         'S': 'Q',
         'C': 'K',
         'K': 'N',
         'N': 'C'}


def prod(l):
    return functools.reduce(lambda x, y: x * y, l)


def rewrite_workload_bounds(src, dst, workload_bounds, model, layer, batchsize, dataflow, phase, terminate, threads, synthetic, sparsity, save, replication, array_width, glb_scaling, dense):  # backward_padding
    w, h, c, n, k, s, r, wpad, hpad, wstride, hstride = workload_bounds
    n = batchsize
    q = int((w - s + 2 * wpad) / wstride) + 1
    p = int((h - r + 2 * hpad) / hstride) + 1

    wu_equiv = k != 'D' and phase == 'wu'
    env_list = {}

    if not wu_equiv:
        print('Workload Dimensions:')
        print('  W        =', w)
        print('  H        =', h)
        print('  C        =', c)
        print('  K        =', k)
        print('  S        =', s)
        print('  R        =', r)
        print('  P        =', p)
        print('  Q        =', q)
        print('  N        =', n)
        print('  W-pad    =', wpad)
        print('  H-pad    =', hpad)
        print('  W-stride =', wstride)
        print('  H-stride =', hstride)
        print()
    else:
        print('Equivalence Test: can we convert WU problem to FW and use cnn-layer.cfg? (at least in the dense case?)')
        print('Workload Dimensions:')
        print('  W        =', w)
        print('  H        =', h)
        print(f'  C <- N {n}')
        print(f'  K <- C {c}')
        print(f'  S <- Q {q}')
        print(f'  R <- P {p}')
        print(f'  P <- R {r}')
        print(f'  Q <- S {s}')
        print(f'  N <- K {k}')
        print('  W-pad    =', wpad)
        print('  H-pad    =', hpad)
        print('  W-stride =', wstride)
        print('  H-stride =', hstride)
        print()
        env_list['TIMELOOP_EQUIVLENT_WU'] = 'True'

    with open(src, "r") as f:
        if "cfg" in src:
            config = libconf.load(f)
        elif "yaml" in src:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    config['problem']['shape'] = shapes[phase]
    if wu_equiv:
        config['problem']['shape'] = shapes['fw']

    if k == 'D':
        depthwise = True
        adapt_depthwise_config(config)
    else:
        depthwise = False
        config['problem']['shape'] += '.yaml'

    if wu_equiv:
        dataflow = convert_dataflow(dataflow)

    if phase == 'wu':
        remove_block_constraint(config)

    if depthwise:
        if dataflow == 'CK':
            dataflow = 'CN'
        dataflow = dataflow.replace('K', 'C')

    rewrite_dataflow(config, dataflow, replication, array_width)

    rewrite_mesh(config, array_width)

    if glb_scaling:
        rewrite_glb_size(config, array_width)

    if not wu_equiv:
        config['problem']['R'] = r
        config['problem']['S'] = s
        config['problem']['P'] = p
        config['problem']['Q'] = q
        config['problem']['C'] = c
        if not depthwise:
            config['problem']['K'] = k
        config['problem']['N'] = n
    else:
        config['problem']['R'] = p
        config['problem']['S'] = q
        config['problem']['P'] = r
        config['problem']['Q'] = s
        config['problem']['C'] = n
        config['problem']['K'] = c
        config['problem']['N'] = k
    config['problem']['Wstride'] = wstride
    config['problem']['Hstride'] = hstride
    config['problem']['Wdilation'] = 1
    config['problem']['Hdilation'] = 1
    config['mapper']['model-name'] = model
    config['mapper']['layer-name'] = layer

    if terminate is not None:
        config['mapper']['victory-condition'] = terminate
    if threads is not None:
        config['mapper']['num-threads'] = threads

    # rewrite synthetic mask configuration
    if not synthetic:
        try:
            config['mapper'].pop('mask-synthetic')
        except KeyError:
            pass
    else:
        config['mapper']['mask-synthetic'] = {}
        if sparsity is not None:
            config['mapper']['mask-synthetic']['target-sparsity'] = sparsity
        if save is not None:
            config['mapper']['mask-synthetic']['synthetic-mask-path'] = save

    if dense:
        opt_metrics = []
        for opt in config['mapper']['optimization-metrics']:
            opt_metrics.append(opt.split('-')[-1])
        config['mapper']['optimization-metrics'] = opt_metrics

    with open(dst, "w") as f:
        if "cfg" in src:
            f.write(libconf.dumps(config))
        elif "yaml" in src:
            f.write(yaml.dump(config))

    return env_list


def convert_dataflow(dataflow):
    pre_convert_dataflow = copy.copy(dataflow)
    converted_dataflow = []
    converted_dataflow.append(wu2fw[pre_convert_dataflow[0]])
    converted_dataflow.append(wu2fw[pre_convert_dataflow[1]])
    converted = ''
    converted = converted.join(converted_dataflow)
    print(f'convert from {dataflow} to {converted}')
    return converted


def remove_block_constraint(config):  # or possibily remove
    for constraint in config['mapspace']['constraints']:
        if constraint['type'] == 'temporal' and constraint['target'] == 'RegFile':
            try:
                constraint.pop('factors')
            except KeyError:
                pass


def rewrite_dataflow(config, dataflow, replication, array_width):
    # loop through constaints, and make sure there is only 1 spatial type constraint
    # dingqing FIXME: not general for more spatial level architecture config
    num_spatial = 0
    for constraint in config['mapspace']['constraints']:
        if num_spatial > 1:
            raise Exception("More than one spatial level! Check the config and the scripts.")
        if constraint['type'] == 'spatial':
            num_spatial += 1

            # determine if it is possible to replicate
            possible2replicate = replication and (not config['problem'][dataflow[0]] > array_width / 2 or not config['problem'][dataflow[1]] > array_width / 2)
            print('possible2replicate?', possible2replicate)
            factors = constraint['factors'].split(' ')
            new_factor = []
            for factor in factors:
                if factor[0] in dataflow:
                    # look at problem size
                    new_factor.append(factor[0] + f'{array_width}')
                elif not possible2replicate:
                    new_factor.append(factor[0] + '1')
            constraint['factors'] = ' '.join(new_factor)

            # rewrite permutation
            # emmmm ugly
            non_spatial_dims = constraint['permutation'].replace(dataflow[0], '').replace(dataflow[1], '')
            constraint['permutation'] = dataflow[0] + non_spatial_dims + dataflow[1]


def rewrite_mesh(config, array_width):
    # honestly, the structure is kinda unnatural...
    pe_subtree = config['architecture']['subtree'][0]['subtree'][0]  # FIXME: this is not generic enough
    pe_name = pe_subtree['name']
    num_pe_prev = re.findall(r'\d+', pe_name)[-1]
    num_pe_new = array_width * array_width - 1
    pe_subtree['name'] = pe_name.replace(num_pe_prev, f'{num_pe_new}')

    # iterate over RF and PE
    for component in pe_subtree['local']:
        component['attributes']['meshX'] = array_width


def rewrite_glb_size(config, array_width):
    scaling_factor = array_width / 16
    # honestly, the structure is kinda unnatural...
    sys_subtree = config['architecture']['subtree'][0]  # FIXME: this is not generic enough
    for comp in sys_subtree['local']:
        if comp['name'] == 'GlobalBuffer':
            comp['attributes']['depth'] = int(comp['attributes']['depth'] * scaling_factor)
            comp['attributes']['n_banks'] = int(comp['attributes']['n_banks'] * scaling_factor)


def adapt_depthwise_config(config):
    config['problem']['shape'] += '-depthwise.yaml'
    try:
        config['problem'].pop('K')
    except KeyError:
        pass
    for constraint in config['mapspace']['constraints']:
        if 'factors' in constraint:
            factors = constraint['factors'].split(' ')
            new_factor = [x for x in factors if x[0] != 'K']
            constraint['factors'] = ' '.join(new_factor)
        if 'permutation' in constraint:
            constraint['permutation'] = ''.join([x for x in constraint['permutation'] if x != 'K'])


def run_timeloop(dirname, configfile, logfile='timeloop.log', env_list={}, dense=False, dense_dirname='dense-timeloop'):
    configfile_path = os.path.join(dirname, os.path.basename(configfile))
    logfile_path = os.path.join(dirname, logfile)

    print('Running timeloop to get mapping')

    def stmt():
        with open(logfile_path, "w") as outfile:
            this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
            if not dense:
                timeloop_executable_location = os.path.join(
                    os.path.dirname(this_file_path), '..', 'build', 'timeloop-mapper')
            else:
                timeloop_executable_location = os.path.join(
                    os.path.dirname(this_file_path), '..', '..', dense_dirname, 'build', 'timeloop-mapper')
            status = subprocess.call([timeloop_executable_location, configfile_path], stdout=outfile, stderr=outfile, env=dict(os.environ, **env_list))
            # status = subprocess.call([timeloop_executable_location, configfile_path, 'ERT.yaml'], stdout=outfile, stderr=outfile)
            if status != 0:
                subprocess.check_call(['cat', logfile_path])
                print('Did you remember to build timeloop and set up your environment properly?')
                sys.exit(1)
    t = timeit.Timer(stmt)
    time = t.timeit(1)
    print('Time to run timeloop = ', time)

    # Move timeloop output files to the right directory
    for f in output_file_names:
        if os.path.exists(f):
            os.rename(f, dirname + '/' + f)
