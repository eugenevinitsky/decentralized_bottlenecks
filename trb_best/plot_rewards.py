import os
import os.path
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
from cycler import cycler
import pandas as pd


# https://gist.github.com/thriveth/8560036
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

mpl.rc('font', size=10)  # controls default text sizes
mpl.rc('axes', titlesize=15)  # fontsize of the axes title
mpl.rc('axes', labelsize=15)  # fontsize of the x and y labels
mpl.rc('xtick', labelsize=13)  # fontsize of the tick labels
mpl.rc('ytick', labelsize=13)  # fontsize of the tick labels
mpl.rc('legend', fontsize=13)  # legend fontsize
mpl.rc('figure', titlesize=20)  # fontsize of the figure title


def init_plt_figure(ylabel, xlabel):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.minorticks_on()
    plt.grid()

def save_plt_figure(title, filename, save_dir='figs', legend_loc='lower right', cols=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Generated', filename)
    plt.legend(loc=legend_loc, ncol=cols)
    plt.ylim(5, 12)
    plt.title(title)
    plt.savefig(fname=os.path.join(save_dir, filename))


rwds = {}

for dirpath, _, files in os.walk('./'):
    for filename in files:
        filepath = os.path.join(dirpath, filename)
        if filepath.endswith('progress.csv'):
            df = pd.read_csv(filepath)
            
            savename = filepath.split('/')[1]

            rwds[savename] = {
                'idx': list(range(len(np.array(df['policy_reward_mean/av'])))),
                'rwd_mean': np.array(df['policy_reward_mean/av']),
                'rwd_min': np.array(df['policy_reward_min/av']),
                'rwd_max': np.array(df['policy_reward_max/av']),
            }

for x in ['complex_agg', 'simple_agg', 'simple_no_agg']:
    init_plt_figure('Reward', 'Iteration')

    for pen in ['0p05', '0p1', '0p2', '0p4']:
        k = f'{x}_{pen}'
        penetration = int(float(k.split('_')[-1].replace('p', '.')) * 100)
        rwd = rwds[k]
        plt.plot(rwd['idx'], rwd['rwd_mean'], linewidth=2, label=f'{penetration}% penetration')
        plt.fill_between(rwd['idx'], rwd['rwd_min'], rwd['rwd_max'], alpha=0.15)

    state_space = x.replace('simple_no_agg', 'Minimal').replace('simple_agg', 'Minimal + Aggregate').replace('complex_agg', 'Radar + Aggregate')
    save_plt_figure(f'Training Iteration vs. Policy Reward for {state_space} State Space', f'reward_{x}.png')
