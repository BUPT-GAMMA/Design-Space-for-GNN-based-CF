import torch
import subprocess
import numpy as np
import pdb
from graphgym.config import cfg
import logging
import gc


def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # nvidia GPU id needs to be remapped to align with the cuda id
    remap = [2, 3, 7, 8, 0, 1, 4, 5, 6, 9]
    return gpu_memory[remap]

def get_total_gpu_memory_map():
    '''Get the total gpu memory.'''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # nvidia GPU id needs to be remapped to align with the cuda id
    remap = [2, 3, 7, 8, 0, 1, 4, 5, 6, 9]
    return gpu_memory[remap]

def auto_select_device(memory_max=6000, memory_bias=200, required_mem_min=5000, strategy='random'):
    '''Auto select GPU device
    memory_max: gpu whose used memory exceeding memory_max will no be random selected
    required_mem_min: min required memory of the program
    '''
    if cfg.device != 'cpu' and torch.cuda.is_available():
        if cfg.device == 'auto':
            total_memory_raw = get_total_gpu_memory_map()
            memory_raw = get_gpu_memory_map()
            available_memory = total_memory_raw - memory_raw
            # set cuda ids which are not available
            unavailable_gpu = []
            for i, m in enumerate(available_memory):
                if m < required_mem_min:
                    unavailable_gpu.append(i)
            
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                #memory_raw[available_gpu] = 2 * memory_max
                cuda = np.argmin(memory_raw)
                available_memory[unavailable_gpu] = 0
                cuda = np.argmax(available_memory)
                logging.info('Total GPU Mem: {}'.format(total_memory_raw))
                logging.info('Available GPU Mem: {}'.format(available_memory))
                logging.info('Unselectable GPU ID: {}'.format(unavailable_gpu))
                logging.info(
                    'Greedy select GPU, select GPU {} with available mem: {}'.format(
                        cuda, available_memory[cuda]))
            elif strategy == 'random':
                # memory = 1 / (memory_raw + memory_bias)
                #memory[memory_raw > memory_max] = 0
                memory = available_memory / available_memory.sum()
                memory[unavailable_gpu] = 0
                gpu_prob = memory / memory.sum()
                np.random.seed()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                np.random.seed(cfg.seed)
                logging.info('Total GPU Mem: {}'.format(total_memory_raw))
                logging.info('Available GPU Mem: {}'.format(available_memory))
                logging.info('Unselectable GPU ID: {}'.format(unavailable_gpu))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.device = 'cuda:{}'.format(cuda)
    else:
        cfg.device = 'cpu'

def collect_unused_memory(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()