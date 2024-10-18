#!/bin/bash
potentials=("zigzag_ridge" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "holder_table" "wavy_plateau" "oakley_ohagan" "sphere" "styblinski_tang")

parallel -j 8 "
    python data_generator.py --potential {} --n-particles 2000 --test-ratio 0.5 &&
    python train.py --solver jkonet-star-potential --dataset potential_{}_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}"