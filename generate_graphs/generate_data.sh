#!/bin/bash

# download graphs data (if no access to S3, generate them using visualize/generate_graphs.py script, cf below)

# regular graphs
aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/graphs_data ./data/
# including alinea_vs_controller (baseline)
aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/alinea_vs_controller ./data/alinea_vs_controller
# graphs trained at random penetration
# aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/graphs_random_pen ./data/
aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/graphs_random_final ./data/


# the following is to generate all graphs on AWS for the following experiments
# already generated, no need to rerun

# simple env - no aggregate - 5% penetration
# simple env - no aggregate - 10% penetration
# simple env - no aggregate - 20% penetration
# simple env - no aggregate - 40% penetration
# simple env - with aggregate - 5% penetration
# simple env - with aggregate - 10% penetration
# simple env - with aggregate - 20% penetration
# simple env - with aggregate - 40% penetration
# complex env - with aggregate - 5% penetration
# complex env - with aggregate - 10% penetration
# complex env - with aggregate - 20% penetration
# complex env - with aggregate - 40% penetration


if false; then
    for exp in 07-26-2020/seedsearch_dqzj2_0p05/seedsearch_dqzj2_0p05/TD3_25_seed=24_2020-07-26_20-03-240trecw9h \
            07-26-2020/seedsearch_dqzj2_0p1/seedsearch_dqzj2_0p1/TD3_30_seed=29_2020-07-26_20-07-11oe2hv4gz \
            07-26-2020/seedsearch_dqzj2_0p2/seedsearch_dqzj2_0p2/TD3_16_seed=15_2020-07-26_20-10-47m7h7gge_ \
            07-26-2020/seedsearch_dqzj2_0p4/seedsearch_dqzj2_0p4/TD3_10_seed=9_2020-07-26_20-14-55k__w7kmt \
            08-14-2020/seedsearch_pos09_0p05/seedsearch_pos09_0p05/TD3_29_seed=28_2020-08-14_08-39-16tmtj4mkr \
            08-14-2020/seedsearch_pos09_0p1/seedsearch_pos09_0p1/TD3_4_seed=3_2020-08-14_08-44-22pa20pu0_ \
            08-14-2020/seedsearch_pos09_0p2/seedsearch_pos09_0p2/TD3_0_seed=None_2020-08-14_08-49-0145nytugu \
            08-14-2020/seedsearch_pos09_0p4/seedsearch_pos09_0p4/TD3_10_seed=9_2020-08-14_08-53-25o0bkk0id \
            07-26-2020/seedsearch_o9di2_0p05/seedsearch_o9di2_0p05/TD3_15_seed=14_2020-07-26_19-50-079dxaty9f \
            07-26-2020/seedsearch_o9di2_0p1/seedsearch_o9di2_0p1/TD3_17_seed=16_2020-07-26_19-53-16cx7riof4 \
            07-26-2020/seedsearch_o9di2_0p2/seedsearch_o9di2_0p2/TD3_18_seed=17_2020-07-26_19-56-22fcx3lomy \
            07-26-2020/seedsearch_o9di2_0p4/seedsearch_o9di2_0p4/TD3_20_seed=19_2020-07-26_19-59-47qydk6h9_
    do
        for pen in 0.05 0.1 0.2 0.4
        do
            echo ${exp} ${pen}
            ray exec ray_autoscale.yaml \
            "python flow/flow/visualize/generate_graphs.py ${exp} 2000 ${pen}" \
            --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
        done
    done
fi


# to download all best policies locally
if false; then
    for exp in 07-26-2020/seedsearch_dqzj2_0p05/seedsearch_dqzj2_0p05/TD3_25_seed=24_2020-07-26_20-03-240trecw9h \
            07-26-2020/seedsearch_dqzj2_0p1/seedsearch_dqzj2_0p1/TD3_30_seed=29_2020-07-26_20-07-11oe2hv4gz \
            07-26-2020/seedsearch_dqzj2_0p2/seedsearch_dqzj2_0p2/TD3_16_seed=15_2020-07-26_20-10-47m7h7gge_ \
            07-26-2020/seedsearch_dqzj2_0p4/seedsearch_dqzj2_0p4/TD3_10_seed=9_2020-07-26_20-14-55k__w7kmt \
            08-14-2020/seedsearch_pos09_0p05/seedsearch_pos09_0p05/TD3_29_seed=28_2020-08-14_08-39-16tmtj4mkr \
            08-14-2020/seedsearch_pos09_0p1/seedsearch_pos09_0p1/TD3_4_seed=3_2020-08-14_08-44-22pa20pu0_ \
            08-14-2020/seedsearch_pos09_0p2/seedsearch_pos09_0p2/TD3_0_seed=None_2020-08-14_08-49-0145nytugu \
            08-14-2020/seedsearch_pos09_0p4/seedsearch_pos09_0p4/TD3_10_seed=9_2020-08-14_08-53-25o0bkk0id \
            07-26-2020/seedsearch_o9di2_0p05/seedsearch_o9di2_0p05/TD3_15_seed=14_2020-07-26_19-50-079dxaty9f \
            07-26-2020/seedsearch_o9di2_0p1/seedsearch_o9di2_0p1/TD3_17_seed=16_2020-07-26_19-53-16cx7riof4 \
            07-26-2020/seedsearch_o9di2_0p2/seedsearch_o9di2_0p2/TD3_18_seed=17_2020-07-26_19-56-22fcx3lomy \
            07-26-2020/seedsearch_o9di2_0p4/seedsearch_o9di2_0p4/TD3_20_seed=19_2020-07-26_19-59-47qydk6h9_
    do
        echo ${exp}

        dir_name=""
        if [[ $exp =~ "dqzj2" ]]; then dir_name="simple_no_agg" fi
        if [[ $exp =~ "pos09" ]]; then dir_name="simple_agg" fi
        if [[ $exp =~ "o9di2" ]]; then dir_name="complex_agg" fi

        if [[ $exp =~ "0p05" ]]; then dir_name="${dir_name}_0p05" fi
        if [[ $exp =~ "0p1" ]]; then dir_name="${dir_name}_0p1" fi
        if [[ $exp =~ "0p2" ]]; then dir_name="${dir_name}_0p2" fi
        if [[ $exp =~ "0p4" ]]; then dir_name="${dir_name}_0p4" fi

        aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/${exp} ~/s3/trb_best/${dir_name}
    done
fi



# generate graphs for best policy trained at random penetration
if false; then
    for exp in 08-22-2020/simple_agg_random_pen_gjdhr/simple_agg_random_pen_gjdhr/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-08-22_23-50-10n21ily9j
    do
        for pen in 0.05 0.1 0.2 0.4
        do
            echo ${exp} ${pen}
            ray exec ray_autoscale.yaml \
            "python flow/flow/visualize/generate_graphs.py ${exp} 2000 ${pen}" \
            --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
        done
    done
fi

# generate graphs for best policy trained at random penetration
gen_graph_universal_controller() {
    for pen in 0.05 0.1 0.2 0.4
    do
        echo exp $2 cp $1 pen ${pen}

        ray exec scripts/ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py $2 $1 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done  
}

gen_graph_universal_controller 2000 09-04-2020/bottleneck_etjza_randompen_complexagg_nolstm/bottleneck_etjza_randompen_complexagg_nolstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-48vg3zcr39
gen_graph_universal_controller 2000 09-04-2020/bottleneck_fodld_randompen_simpleagg_lstm/bottleneck_fodld_randompen_simpleagg_lstm/TD3_14_actor_lr=0.001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-39kow4g_f4
gen_graph_universal_controller 2000 09-04-2020/bottleneck_nrzod_randompen_simplenoagg_lstm/bottleneck_nrzod_randompen_simplenoagg_lstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-40fd5lzgmm
gen_graph_universal_controller 2000 09-04-2020/bottleneck_pxanz_randompen_complexagg_lstm/bottleneck_pxanz_randompen_complexagg_lstm/TD3_1_actor_lr=0.0001,critic_lr=0.001,n_step=1,prioritized_replay=True_2020-09-05_01-06-37b4zv0_0n
gen_graph_universal_controller 2000 09-04-2020/bottleneck_riiod_randompen_simpleagg_nolstm/bottleneck_riiod_randompen_simpleagg_nolstm/TD3_4_actor_lr=0.001,critic_lr=0.001,n_step=5,prioritized_replay=True_2020-09-05_01-06-38ob13w45p
gen_graph_universal_controller 2000 09-04-2020/bottleneck_msdor_randompen_simplenoagg_nolstm/bottleneck_msdor_randompen_simplenoagg_nolstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-41d0q8ie4m

gen_graph_universal_controller 800 09-04-2020/bottleneck_etjza_randompen_complexagg_nolstm/bottleneck_etjza_randompen_complexagg_nolstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-48vg3zcr39
gen_graph_universal_controller 800 09-04-2020/bottleneck_pxanz_randompen_complexagg_lstm/bottleneck_pxanz_randompen_complexagg_lstm/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-38y90x5yq1
gen_graph_universal_controller 1600 09-04-2020/bottleneck_fodld_randompen_simpleagg_lstm/bottleneck_fodld_randompen_simpleagg_lstm/TD3_5_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=True_2020-09-05_01-06-39j51sapfu
gen_graph_universal_controller 1600 09-04-2020/bottleneck_nrzod_randompen_simplenoagg_lstm/bottleneck_nrzod_randompen_simplenoagg_lstm/TD3_12_actor_lr=0.001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-40vb748u5w
gen_graph_universal_controller 1200 09-04-2020/bottleneck_riiod_randompen_simpleagg_nolstm/bottleneck_riiod_randompen_simpleagg_nolstm/TD3_9_actor_lr=0.0001,critic_lr=0.001,n_step=1,prioritized_replay=False_2020-09-05_01-06-39achysxl7
gen_graph_universal_controller 400 09-04-2020/bottleneck_msdor_randompen_simplenoagg_nolstm/bottleneck_msdor_randompen_simplenoagg_nolstm/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-41jc1ppsvu

## took TD13 (see below) for simple agg in the end


# tmp graphs random not converged
for exp in 09-04-2020/bottleneck_etjza_randompen_complexagg_nolstm/bottleneck_etjza_randompen_complexagg_nolstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-48vg3zcr39 \
           09-04-2020/bottleneck_fodld_randompen_simpleagg_lstm/bottleneck_fodld_randompen_simpleagg_lstm/TD3_5_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=True_2020-09-05_01-06-39j51sapfu \
           09-04-2020/bottleneck_nrzod_randompen_simplenoagg_lstm/bottleneck_nrzod_randompen_simplenoagg_lstm/TD3_12_actor_lr=0.001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-40vb748u5w \
           09-04-2020/bottleneck_pxanz_randompen_complexagg_lstm/bottleneck_pxanz_randompen_complexagg_lstm/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-38y90x5yq1
do
    for pen in 0.05 0.1 0.2 0.4
    do
        echo ${exp} ${pen}
        ray exec ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py ${exp} 800 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done
done

for exp in 09-04-2020/bottleneck_riiod_randompen_simpleagg_nolstm/bottleneck_riiod_randompen_simpleagg_nolstm/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-39a820ojqf \
           09-04-2020/bottleneck_msdor_randompen_simplenoagg_nolstm/bottleneck_msdor_randompen_simplenoagg_nolstm/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-05_01-06-41jc1ppsvu
do
    for pen in 0.05 0.1 0.2 0.4
    do
        echo ${exp} ${pen}
        ray exec ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py ${exp} 400 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done
done



# try fine tuning
gen_graph_universal_controller() {
    for pen in 0.05 0.1 0.2 0.4
    do
        echo exp $2 cp $1 pen ${pen}

        ray exec scripts/ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py $2 $1 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done  
}

gen_graph_universal_controller 588 09-07-2020/bottleneck_abcde_randompen_simpleagg_nolstm_finetune/bottleneck_abcde_randompen_simpleagg_nolstm_finetune/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-07_19-44-25nhu_qfh2
gen_graph_universal_controller 686 09-07-2020/bottleneck_abcde_randompen_simpleagg_nolstm_finetune/bottleneck_abcde_randompen_simpleagg_nolstm_finetune/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-07_19-44-25nhu_qfh2




# reduced radar env & lane change
for exp in 07-26-2020/seedsearch_o9di2_0p05/seedsearch_o9di2_0p05/TD3_15_seed=14_2020-07-26_19-50-079dxaty9f \
        07-26-2020/seedsearch_o9di2_0p1/seedsearch_o9di2_0p1/TD3_17_seed=16_2020-07-26_19-53-16cx7riof4 \
        07-26-2020/seedsearch_o9di2_0p2/seedsearch_o9di2_0p2/TD3_18_seed=17_2020-07-26_19-56-22fcx3lomy \
        07-26-2020/seedsearch_o9di2_0p4/seedsearch_o9di2_0p4/TD3_20_seed=19_2020-07-26_19-59-47qydk6h9_
do
    for pen in 0.05 0.1 0.2 0.4
    do
        echo ${exp} ${pen}
        ray exec scripts/ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py ${exp} 2000 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done
done



# no congest nb
for exp in 09-03-2020/bottleneck_pseot_cna_0p1_nc_nada/bottleneck_pseot_cna_0p1_nc_nada/TD3_13_actor_lr=0.0001,critic_lr=0.001,n_step=5,prioritized_replay=False_2020-09-03_23-17-08jclcpt7i
do
    for pen in 0.1
    do
        for cp in 400 800 1200 1600 2000
        do
            echo ${exp} ${pen} ${cp}
            ray exec scripts/ray_autoscale.yaml \
            "python flow/flow/visualize/generate_graphs.py ${exp} ${cp} ${pen}" \
            --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
        done
    done
done




# reduced radar env & lane change, without extra
gen_graphs() {
    echo exp $1 pen $2
    ray exec scripts/ray_autoscale.yaml \
    "python flow/flow/visualize/generate_graphs.py $1 2000 $2" \
    --start --stop --tmux --cluster-name nathan_graphs_$2_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
}

for i in 1 2
do
    gen_graphs 07-26-2020/seedsearch_o9di2_0p05/seedsearch_o9di2_0p05/TD3_15_seed=14_2020-07-26_19-50-079dxaty9f 0.05
    gen_graphs 07-26-2020/seedsearch_o9di2_0p1/seedsearch_o9di2_0p1/TD3_17_seed=16_2020-07-26_19-53-16cx7riof4 0.1
    gen_graphs 07-26-2020/seedsearch_o9di2_0p2/seedsearch_o9di2_0p2/TD3_18_seed=17_2020-07-26_19-56-22fcx3lomy 0.2
    gen_graphs 07-26-2020/seedsearch_o9di2_0p4/seedsearch_o9di2_0p4/TD3_20_seed=19_2020-07-26_19-59-47qydk6h9_ 0.4
done


# reduced radar or lane change on universal
for exp in 09-04-2020/bottleneck_etjza_randompen_complexagg_nolstm/bottleneck_etjza_randompen_complexagg_nolstm/TD3_15_actor_lr=0.0001,critic_lr=0.0001,n_step=5,prioritized_replay=False_2020-09-05_01-06-48vg3zcr39
do
    for pen in 0.05 0.1 0.2 0.4
    do
        echo ${exp} ${pen}
        ray exec scripts/ray_autoscale.yaml \
        "python flow/flow/visualize/generate_graphs.py ${exp} 2000 ${pen}" \
        --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
    done
done
