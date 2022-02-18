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

from TSPTrainer import TSPTrainer as Trainer

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
parser.add_argument('--train_episodes', type=int, default=100*1000)
parser.add_argument('--train_batch_size', type=int, default=20)
parser.add_argument('--desc', type=str, default='train__tsp_n20')


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

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': args.epoch,
    'train_episodes': args.train_episodes,
    'train_batch_size': 20,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_' + str(args.problem_size) + '.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': args.desc,
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
