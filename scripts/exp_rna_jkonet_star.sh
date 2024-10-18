python preprocess_rna_seq.py --n-components 5
python data_generator.py --load-from-file RNA_PCA_5 --test-ratio 0.4  --split-population

parallel -j 5 python train.py --dataset RNA_PCA_5 --solver jkonet-star-time-potential --seed {} --wandb --epochs 100 ::: 0 1 2 3 4