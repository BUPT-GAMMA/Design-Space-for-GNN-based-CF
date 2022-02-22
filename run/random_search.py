import random
import argparse
import os

from graphgym.utils import io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        dest='input_dir',
        help='directory of candidate configurations',
        default='configs/rec_pointwise_condense_grid_rec_pointwise_condense',
        type=str
    )
    parser.add_argument(
        '--sample_num',
        dest='sample_num',
        help='Number of random samples',
        default=5,
        type=int
    )
    parser.add_argument(
        '--repeat',
        dest='repeat',
        help='Repeated times of random sampling',
        default=3,
        type=int
    )
    return parser.parse_args()
    

args = parse_args()
input_dir = args.input_dir
sample_num = args.sample_num
output_parent_dir = f'{input_dir}/random/{sample_num}'
io.makedirs_rm_exist(output_parent_dir)
for i in range(args.repeat):
    seed = i + 1
    random.seed(seed)
    all_configs = os.listdir(input_dir)
    sampled_configs = random.sample(all_configs, sample_num)
    output_dir = f'{output_parent_dir}/{seed}'
    io.makedirs(output_dir)
    for config in sampled_configs:
        os.system(f'cp {input_dir}/{config} {output_dir}')
    print(f'Succsessfully randomly sampled {len(sampled_configs)} configurations!')

