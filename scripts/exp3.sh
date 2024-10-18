#!/bin/bash
potentials=("wavy_plateau" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "oakley_ohagan" "zigzag_ridge" "holder_table")
interactions=("wavy_plateau" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "oakley_ohagan" "zigzag_ridge")
betas=(0.0 0.1 0.2)

export potentials interactions betas

parallel -j 8 "
    potential={1}; interaction={2}; beta={3};
    python data_generator.py --potential \$potential --interaction \$interaction --n-particles 2000 --test-ratio 0.5 --internal wiener --beta \$beta &&
    python train.py --solver jkonet-star --dataset potential_\${potential}_internal_wiener_beta_\${beta}_interaction_\${interaction}_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb &&
    python train.py --solver jkonet-star-linear --dataset potential_\${potential}_internal_wiener_beta_\${beta}_interaction_\${interaction}_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}" ::: "${interactions[@]}" ::: "${betas[@]}"