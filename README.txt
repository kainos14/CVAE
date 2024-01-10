To run the framework in command line : python TempSSGAN.py 128 300 PAMAP2

Folder structure to run the model:

������ preprocessing.py # Preprocessing file.
������ TempSSGAN.py # File containing networks and training.
������ PAMAP2_Dataset

PAMAP2 dataset is available at https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring.

Framework uses Tensorflow 2+, tensorflow_addons, numpy, pandas, matplotlib, scikit-learn.