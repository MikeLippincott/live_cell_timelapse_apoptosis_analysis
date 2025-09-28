#!/bin/bash

conda activate timelapse_analysis_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit 1

python 0.data_splits.py
python 1.train_model.py
python 2.evaluate_model.py
python 3.prediction.py

conda deactivate
conda activate R_timelapse_env
Rscript 4.plot_results.r
Rscript 5.plot_model_peformances.r

cd .. || exit 1
conda deactivate

echo "Timelapse model training and evaluation completed"
