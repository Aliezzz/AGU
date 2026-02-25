# Adaptive Graph Unlearning

**Requirements**

Hardware environment: Intel(R) Xeon(R) 8352V @ 2.10GHz, NVIDIA GeForce RTX 4090 with 24GB.

Software environment: Ubuntu 20.04.5, Python 3.10, Pytorch 2.1.2, and CUDA 11.8.0.
  1. Please refer to PyTorch and PyG to install the environments;
  
  2. Run 'pip install -r requirements.txt' to download required packages;

**Training**

To train model(s) in the paper
  1. Please refer to the configs folds to modify the hyperparameters

     config.py - Setting the path for data loading and saving.

     parameter_parser.py - Parameter settings for modeling and training.

  2. Open main.py to start unlearning

     We provide Cora dataset as example.

     Meanwhile, you can personalize your settings (lib_dataset/lib_gnn_model/exp).

     Run this command:

```
python main.py
```
