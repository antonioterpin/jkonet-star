#!/bin/bash
# "zigzag_ridge" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "holder_table" "oakley_ohagan" "sphere" 0.01 0.1 
potentials=("wavy_plateau" "styblinski_tang")
epsilon=(1.0)

parallel -j 8 "
    potential={1} epsilon={2}
    python data_generator.py --potential \${potential} --n-particles 2000 --test-ratio 0.5 --sinkhorn \${epsilon} &&
    python train.py --solver jkonet-star-potential --dataset potential_\${potential}_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_\${epsilon} --wandb
" ::: "${potentials[@]}" ::: "${epsilon[@]}"