# LRP-based Structured Pruning

## How to run

### Installation
- Clone this repository and enter it: 
    ```bash
    git clone https://github.com/LukasHedegaard/LRP_pruning.git
    cd LRP_pruning
    ```
- (Optionally) create conda environment:
    ```bash
    conda create --name LRP_pruning python=3.10
    ```
- Install as editable module
    ```bash
    pip install -e .[dev]
    ```

### Pretrained weights
Trained model weights are available [here](https://drive.google.com/drive/folders/1m6aV5Zv8tAytvxF6qY4m9nyqlkKv0y72?usp=sharing).


### Lint
Lint using [black](https://github.com/psf/black) and [isort](https://github.com/timothycrosley/isort/):
```bash
make lint
```

## Model
> ResNet-18, ResNet-50\
> VGG-16, AlexNet
> 

## Toy Experiment
>  https://github.com/seulkiyeom/LRP_Pruning_toy_example
>  

## Reference 
```bibtex
@article{yeom2021pruning,
  title={Pruning by explaining: A novel criterion for deep neural network pruning},
  author={Yeom, Seul-Ki and
          Seegerer, Philipp and
          Lapuschkin, Sebastian and
          Binder, Alexander and
          Wiedemann, Simon and
          M{\"u}ller, Klaus-Robert and
          Samek, Wojciech},
  journal={Pattern Recognition},
  pages={107899},
  year={2021},
  publisher={Elsevier}
}
```
