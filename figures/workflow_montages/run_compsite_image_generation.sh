#!/bin/bash

# Run this script to generate composite images and psuedo colored images for all well_fovs and timepoints
conda activate timelapse_analysis_env

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb
cd scripts || echo "Error: scripts directory not found" >&2; exit 1;

python get_psuedo_colored_images.py --composite_or_single_channel False
# False is for single channel psuedo colored images
# True is for composite images

cd ../ || exit

conda deactivate

echo "Done running get_psuedo_colored_images.py"
