import os
import subprocess
from flow.visualize.bottleneck_results import run_bottleneck_results
import ray

EXPERIMENTS_PATH = os.path.expanduser('~/experiments')
OUTPUT_PATH = os.path.expanduser('~/bottleneck_graphs')

p1 = subprocess.Popen(
    'aws s3 sync {} {} {}'.format(
        EXPERIMENTS_PATH,
        's3://nathan.experiments/trb_bottleneck_paper/06-28-2020',
        '--exclude="*" --include="*.pkl" --include="*checkpoint*" --include="*event*'))
p1.wait(60000)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

ray.init()

for (dirpath, dirnames, filenames) in os.walk(EXPERIMENTS_PATH):
    if 'checkpoint_350' in dirnames:   
        run_bottleneck_results(
            outflow_min=400, outflow_max=3600, step=100, num_trials=20,
            output_path=OUTPUT_PATH, filename=dirpath,
            checkpoint_dir=dirpath, checkpoint_num=350,
            gen_emission=False, render_mode='no_render',
            horizon=400, end_len=500)

for i in range(4):
    try:
        p1 = subprocess.Popen(
            'aws s3 sync {} {}'.format(
                OUTPUT_PATH,
                's3://nathan.experiments/trb_bottleneck_paper/graphs_test'))
        p1.wait(60000)
    except Exception as e:
        print('This is the error ', e)