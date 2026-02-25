# Adaptive Graph Unlearning (AGU)

**This repository is the official implementation of [Adaptive Graph Unlearning](https://www.ijcai.org/proceedings/2025/0308.pdf), published at IJCAI 2025.**

---

## Citation

If you use this work or code in your research, please cite us:

```bibtex
@inproceedings{ding2025adaptive,
  title     = {Adaptive Graph Unlearning},
  author    = {Ding, Pengfei and Wang, Yan and Liu, Guanfeng and Zhu, Jiajie},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI} 2025},
  year      = {2025},
  url       = {https://www.ijcai.org/proceedings/2025/0308.pdf}
}
```

**Paper:** [Adaptive Graph Unlearning](https://www.ijcai.org/proceedings/2025/0308.pdf) (IJCAI 2025)  
**Authors:** Pengfei Ding, Yan Wang, Guanfeng Liu, Jiajie Zhu (Macquarie University)

---

## Requirements

**Hardware:** Intel(R) Xeon(R) 8352V @ 2.10GHz, NVIDIA GeForce RTX 4090 24GB

**Software:** Ubuntu 20.04.5, Python 3.10, PyTorch 2.1.2, CUDA 11.8.0

1. Please refer to [PyTorch](https://pytorch.org/) and [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) for environment setup.
2. Run `pip install -r requirements.txt` to install the required packages.

---

## Training

To reproduce the model(s) in the paper:

1. **Configure hyperparameters** (see `configs` and the following files):
   - `config.py` — paths for data loading and saving
   - `parameter_parser.py` — model and training parameters

2. **Run unlearning**
   - Start from `main.py`; we provide Cora as an example dataset.
   - You can customize `lib_dataset`, `lib_gnn_model`, and `exp` as needed.

```bash
python main.py
```
