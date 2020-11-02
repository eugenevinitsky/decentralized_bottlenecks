import os
import ray
from flow.visualize.bottleneck_results import run_bottleneck_results
import subprocess
import errno
import sys


def aws_sync(src, dest):
    print('AWS S3 SYNC FROM >>> {} <<< TO >>> {} <<<'.format(src, dest))
    for _ in range(4):
        try:
            p1 = subprocess.Popen('aws s3 sync {} {}'.format(src, dest).split(' '))
            p1.wait(3600)
        except Exception as e:
            print('This is the error ', e)


NUM_TEST_TRIALS = 20

OUTFLOW_MIN = 400
OUTFLOW_MAX = 3600
OUTFLOW_STEP = 100

if __name__ == '__main__':
    
    
    if len(sys.argv) <= 3:
        print('usage: generate_graphs.py exp_cp_path cp_number [evaluation penetration]')
    exp_cp_path = sys.argv[1]
    cp = sys.argv[2]
    if len(sys.argv) >= 4:
        penetration = float(sys.argv[3])
    else:
        penetration = None  # keep penetration used during training

    print("exp cp path: ", exp_cp_path)
    print("cp: ", cp)
    print("penetration: ", penetration)
    exp_title_lst = exp_cp_path.split('/')
    exp_title = '/'.join([exp_title_lst[0]] + exp_title_lst[2:])
    exp_title = exp_title.replace('/', '_') + "_CP_" + str(cp)
    if penetration is not None:
        exp_title += f'_PEN_{penetration}'

    # download checkpoints from AWS
    try:
        os.makedirs(os.path.expanduser("~/ray_results"))
    except:
        pass
    aws_sync('s3://nathan.experiments/trb_bottleneck_paper/' + exp_cp_path,
             os.path.expanduser("~/ray_results/trb_bottleneck_paper/" + exp_cp_path))

    ray.init(num_cpus=35)
    
    output_path = os.path.join(os.path.expanduser('~/bottleneck_results'))
    try:
        os.makedirs(output_path)
    except:
        pass

    local_cp_path = os.path.join(os.path.expanduser("~/ray_results/trb_bottleneck_paper"), exp_cp_path)
    run_bottleneck_results(OUTFLOW_MIN, OUTFLOW_MAX, OUTFLOW_STEP, NUM_TEST_TRIALS, output_path, exp_title, local_cp_path,
                            gen_emission=False, render_mode='no_render', checkpoint_num=str(cp),
                            horizon=400, end_len=500, penetration=penetration)  

    aws_sync(output_path,
            os.path.join("s3://nathan.experiments/trb_bottleneck_paper/graphs/", exp_title))
