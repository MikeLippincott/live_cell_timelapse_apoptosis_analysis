#!/bin/bash

# this script is used to run the EDA process

# activate the conda environment
conda activate timelapse_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts notebooks/*.ipynb

cd scripts || exit

# run the EDA script
python 0.generate_umap_embeddings.py --data_mode "CP"
python 0.generate_umap_embeddings.py --data_mode "combined"

# deactivate the conda environment
conda deactivate

conda activate R_timelapse_env

Rscript 1.visualize_umaps.r --data_mode "CP"
Rscript 1.visualize_umaps.r --data_mode "combined"

conda deactivate

cd ../ || exit

# end of script
echo "EDA process completed"
