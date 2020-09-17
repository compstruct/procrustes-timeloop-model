#! /usr/bin/env python3

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

import inspect
import os
# from pathlib import Path
import pprint
import subprocess
import sys
import xml.etree.ElementTree as ET
import argparse

import numpy as np
import libconf
import yaml
import json
from distutils.dir_util import copy_tree

from common import *
from utils import *
import timeloop
import parse_timeloop_output


def main():
    parser = argparse.ArgumentParser(description='Run Timeloop')
    parser.add_argument('--config', '-c', default='../configs/mapper/VGGN.yaml', help='config file')  # yaml
    parser.add_argument('--log', '-l', default='timeloop.log', help='name of log file')
    parser.add_argument('--output_dir', '-o', default='results', help='name of log file')
    parser.add_argument('--fake_eval', nargs='?', type=str2bool, const=True,
                        default=False, help='test evaluation that no actual evalution is performed')

    # search configuration
    parser.add_argument('--net', default='vgg', choices=['vgg', 'wrn', 'dense', 'resnet18', 'mobilenetv2'], help='model name')
    parser.add_argument('--dataset', default='cifar-10', choices=['cifar-10', 'imagenet'], help='dataset name')
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='batchsize of the problem')
    parser.add_argument('--dataflow', default='CN', choices=['CK', 'CN', 'KN', 'CP', 'KP', 'PQ', 'PN'], help='spatial mapping')
    parser.add_argument('--phase', default='fw', choices=['fw', 'bw', 'wu'], help='Training phase')  # , 'wu'

    # mapper configuration
    parser.add_argument('--terminate', nargs='?', type=int, const=1000, help='termination condition: number of consecutive suboptimal valid mapping found')
    parser.add_argument('--threads', nargs='?', type=int, const=32, help='number of threads used for mapping search')
    # synthetic mask
    parser.add_argument('--synthetic', nargs='?', type=str2bool, const=True,
                        default=False, help='Is data mask synthetic?')
    parser.add_argument('--sparsity', nargs='?', type=float, const=0.1, help='synthetic sparsity of the problem')
    parser.add_argument('--act_sparsity', default='_act_sparsity.json', help='file suffix for activation sparsity')
    # usually let's not provide this flag to save some space?
    parser.add_argument('--save', nargs='?', type=str, const='saved_synthetic_mask', help='name of saved synthetic mask')

    # naive replication
    parser.add_argument('--replication', nargs='?', type=str2bool, const=True, default=False, help='do we apply naive replication?')

    # scability exp
    parser.add_argument('--array_width', type=int, default=16, help='PE array width')
    parser.add_argument('--glb_scaling', nargs='?', type=str2bool, const=True, default=False, help='scale GLB based on array_width')

    # Evaluate Dense Timeloop?
    parser.add_argument('--dense', nargs='?', type=str2bool, const=True,
                        default=False, help='evaluate use original timeloop')
    parser.add_argument('--dense_dirname', default='dense-timeloop', help='directory name of dense timeloop')
    args = parser.parse_args()

    print(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # load yaml

    use_act_sparsity = False
    if args.phase == 'wu' and not args.dense:
        with open(args.net + args.act_sparsity) as f:
            act_sparsity_data = json.load(f)
            use_act_sparsity = True

    target_layers = layer_sizes[args.dataset][args.net]
    target_names = layer_names[args.dataset][args.net]

    total_cycles = 0
    total_energy = 0
    for i in range(0, len(target_names)):
        problem = target_layers[i]
        name = target_names[i]

        # TODO: redesign path
        # make configuration first, layer last to facilitate potential overall speedup and energy saving

        # training phase, batchsize, network, and layer
        dirname = args.output_dir + f'/{args.array_width}_{args.glb_scaling}/{args.phase}_{args.batchsize}/{args.dataset}_{args.net}_{name}/'

        # which source accelerator we use and spatial dataflow
        dirname += ('dense' if args.dense else 'sparse') + '_{}'.format(args.dataflow) + ('_replicate/' if args.replication else '/')

        # synthetic (with target sparsity) or acutal mask
        dirname += 'synthetic_{}/'.format(args.sparsity) if args.synthetic else 'actual/'

        subprocess.check_call(['mkdir', '-p', dirname])
        print('Problem {} result dir: {}'.format(i, dirname))  # use this to hint the problem we are working on

        if args.dense:
            if os.path.isfile(dirname + 'timeloop-mapper.stats.txt'):
                print('The current dense problem evaluated already, skip!')
                continue
            if problem in target_layers[:i]:
                j = target_layers.index(problem)  # repeated index
                print('Same Config as Problem {} layer {}, skip evaluation and copy result'.format(j, target_names[j]))
                src_dirname = args.output_dir + f'/{args.array_width}_{args.glb_scaling}/{args.phase}_{args.batchsize}/{args.dataset}_{args.net}_{target_names[j]}/'
                # which source accelerator we use and spatial dataflow
                src_dirname += ('dense' if args.dense else 'sparse') + '_{}'.format(args.dataflow) + ('_replicate/' if args.replication else '/')
                # synthetic (with target sparsity) or acutal mask
                src_dirname += 'synthetic_{}/'.format(args.sparsity) if args.synthetic else 'actual/'
                copy_tree(src_dirname, dirname)
                continue
        else:
            if os.path.isfile(dirname + 'timeloop-mapper.stats.txt'):
                print('The current sparse problem evaluated already, skip!')
                continue
        # dump the all configuration to check, also hopefully helps
        # reproducibility
        with open(os.path.join(dirname, 'args.json'), 'w') as arg_log:
            json.dump(vars(args), arg_log)

        # (Optional): Adapt to Path module
        env_list = timeloop.rewrite_workload_bounds(
            src=args.config,
            dst=os.path.join(dirname, os.path.basename(args.config)),
            workload_bounds=problem,
            model=args.net,  # for actual mask only
            # datset = args.dataset, # unimplemented
            layer=name,  # for actual mask only
            batchsize=args.batchsize,
            dataflow=args.dataflow,
            phase=args.phase,
            terminate=args.terminate,
            threads=args.threads,
            synthetic=True if use_act_sparsity else args.synthetic,
            sparsity=act_sparsity_data[name] if use_act_sparsity else args.sparsity,
            save=args.save,
            replication=args.replication,
            array_width=args.array_width,
            glb_scaling=args.glb_scaling,
            dense=args.dense)

        if args.fake_eval:
            print('in fake eval mode, the problem will be evaluate in actual run')
            continue

        timeloop.run_timeloop(
            dirname=dirname,
            configfile=args.config,
            logfile=args.log,
            env_list=env_list,
            dense=args.dense,
            dense_dirname=args.dense_dirname)

        cycle, energy, mac = parse_timeloop_output.parse_timeloop_stats(dirname, args.dense)
        if energy == {}:
            print("Timeloop couldn't find a mapping for this problem within the search parameters, please check the log for more details.")
        else:
            print("Run successful, see log for text stats, or use the Python parser to parse the XML stats.")

    print("DONE.")


if __name__ == '__main__':
    main()
