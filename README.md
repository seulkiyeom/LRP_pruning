# LRP-based Structured Pruning

<div align="left">
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img src="https://img.shields.io/static/v1?label=License&message=CC-BY-NC-SA 4.0&color=blue" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

## Installation
- Clone this repository and enter it: 
    ```bash
    git clone https://github.com/LukasHedegaard/LRP_pruning.git
    cd LRP_pruning
    ```
- (Optionally) create conda environment:
    ```bash
    conda env create --name LRP_pruning python=3.10
    ```
- Install as editable module
    ```bash
    pip install -e .[dev]
    ```

## Run training + pruning

### Layer-wise Relevance Propagation Pruning 
_Yeom et al., "Pruning by explaining: A novel criterion for deep neural network pruning", in: Pattern Recognition 2021_
```bash
python main.py --train --prune --method-type lrp \\
    --arch resnet50 --lr 0.0025 --batch-size 64 --epochs 20 --recovery-epochs 20
```
### Grad Pruning + L2 norm 
_X. Sun et al., "Meprop: sparsified back propagation for accelerated deep learning with reduced overfitting", in: International Conference on Machine Learning (ICML), 2017_
```bash
python main.py --train --prune --method-type grad --norm
```

### Taylor Pruning + L2 Norm 
_P. Molchanov et al., "Pruning convolutional neural networks for resource efficient transfer learning", in: International Conference on Learning Representations (ICLR), 2017_
```bash
python main.py --train --prune --method-type grad --norm
```

### Weight Pruning + L2 norm
```bash
python main.py --train --prune --method-type weight --norm
```

### Training options
```bash
python main.py --help
```


## Lint
Lint using [black](https://github.com/psf/black) and [isort](https://github.com/timothycrosley/isort/):
```bash
make lint
```

