# Deep-Learning-GANs
Generative Adversarial Networks (GAN) and Deep Convolutional GAN (DCGAN) for MNIST and Cifar10 datasets

## Requirements
- python3
- pytorch and torchvision
```bash
pip install torch torchvision
```
    
## Usage
The default setting is implemented GCGAN for MNIST dataset. The command is as follows:
```bash
python3 main.py
```
Also, the arguments can be used as follows:
```bash
python3 main.py -b 100 -e 20 -gt gan -d cifar10
```
- -b: batch size
- -e: number of epochs
- -gt: gan type (gan or gcgan)
- -d: dataset (mnist or cifar10)
