# Deep-Learning-GANs
The Python program using PyTorch for implementing Generative Adversarial Networks (GAN) and Deep Convolutional GAN (DCGAN) for MNIST, Cifar10 and STL10 datasets

## Requirements
- python3
- pytorch and torchvision
```bash
pip install torch torchvision
```
    
## Usage
The default setting is
- batch size: 200
- number of epochs: 100
- GAN's type: DCGAN
- dataset: MNIST

The command is as follows:
```bash
python3 main.py
```
Also, the arguments can be used as follows:
```bash
python3 main.py -b 100 -e 20 -gt gan -d cifar10
```
- -b: batch size
- -e: number of epochs
- -gt: gan type (gan or dcgan)
- -d: dataset (mnist, cifar10 and stl10)

## References
- [Ian J. Goodfellow, et al, Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Alec Radford, et al, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Keiichiro Miyamoto, Yohei Okawa and Takuya Mouri, PyTorchニューラルネットワーク実装ハンドブック (Japanese book)](https://www.amazon.co.jp/PyTorchニューラルネットワーク実装ハンドブック-Pythonライブラリ定番セレクション-宮本-圭一郎/dp/4798055476/ref=sr_1_1?ie=UTF8&qid=1547369586&sr=8-1&keywords=pytorch)
