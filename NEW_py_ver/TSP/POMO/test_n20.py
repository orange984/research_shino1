##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester


import numpy as np
import torch
import argparse


parser = argparse.ArgumentParser(description='Description of the argument')

parser.add_argument('--DEBUG_MODE', action='store_true')
parser.add_argument('--problem_size', type=int, default=20)
parser.add_argument('--pomo_size', type=int, default=1)
parser.add_argument('--path', type=str, default='./result/saved_tsp20_model')
parser.add_argument('--epoch', type=int, default=510)
parser.add_argument('--NORM_MODE', action='store_true')
parser.add_argument('--TEST_MODE', action='store_true')
parser.add_argument('--test_set', type=str,
                    default='../TSProblem/testset_n20.npy')
parser.add_argument('--test_episodes', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=10)
parser.add_argument('--augmentation_enable', action='store_true')
parser.add_argument('--aug_factor', type=int, default=8)
parser.add_argument('--aug_batch_size', type=int, default=10)
parser.add_argument('--desc', type=str, default='test__tsp_n20')


args = parser.parse_args()

##########################################################################################
# Machine Environment Config

DEBUG_MODE = args.DEBUG_MODE
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# parameters
env_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'problem_size': args.problem_size,
    'pomo_size': args.pomo_size,
    'NORM_MODE': args.NORM_MODE,
    'TEST_MODE': args.TEST_MODE,
    'test_set': args.test_set,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        # directory path of pre-trained model and log files saved.
        'path': args.path,
        'epoch': args.epoch,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': args.test_episodes,
    'test_batch_size': args.test_batch_size,
    'augmentation_enable': args.augmentation_enable,
    'aug_factor': args.aug_factor,
    'aug_batch_size': args.aug_batch_size,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': args.desc,
        'filename': 'run_log'
    }
}

##########################################################################################
# main


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if DEBUG_MODE:
        _set_debug_mode()
    print(np.load(args.test_set)[0][0][:5])
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10000


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(
        USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key]))
     for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################
if __name__ == "__main__":
    main()
