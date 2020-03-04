## Classification of nail images
Automatic classification of nails

This repo contains
- nail_classification.ipynb showing preprocessing steps and model architecture.
- functions.py with utility functions.
- app.py which is the flask script for the api.

### Pre-processing
The data is pre-processed with the following steps:
- Cropping
- Converting to grayscale
- Patch around object of interest is extracted

### Data augmentation
A big challenge here is the size of the data set. With only 100 samples per class it is difficult to train a general model. To increase the size of the data set, offline data augementation is performed: Horizontal and vertical flips as well as rotations in 90, 180, and 270 degrees.

### Model fitting
- The classification is done with a CNN model consisting of two convolutional layers, one dense layer, and some dropout and batch normalization.
- The model achieved a test accuracy of 81%.
