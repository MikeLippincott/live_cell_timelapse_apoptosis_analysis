# Timelapse model
This model is trained to predict the AnnexinV feature space in the terminal staining.
The data_splitting strategy is to train on the final timepoint and test on the previous time points.
The model will be trained on the two wells from each dose and tested on the other wells (1).

The model is evaluated on predicting the terminal profiles from each of the previous time points.

Note: The terminal profiles stain for DNA and AnnexinV.
