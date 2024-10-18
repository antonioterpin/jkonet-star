potentials=("wavy_plateau" "oakley_ohagan" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "zigzag_ridge" "holder_table")
dims=(10 20 30 40 50)
n_particles_values=(2000 5000 10000 15000 20000)

export potentials dims n_particles_values

parallel -j 8 "
    potential={1}; dim={2}; n_particles={3};
    python data_generator.py --potential \$potential --n-particles \$n_particles --test-ratio 0.5 --dimension \$dim &&
    python train.py --solver jkonet-star-potential --dataset potential_\${potential}_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_\${dim}_N_\${n_particles}_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}" ::: "${dims[@]}" ::: "${n_particles_values[@]}"
