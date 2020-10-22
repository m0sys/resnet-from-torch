<h1 align="center">Welcome to resnet-from-torch 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>


> Implementation of ResNet34 & ResNet50 networks using PyTorch as the base.

## Install

```sh
conda env create -f environment.yml
```

## Results

- Performance comparable to PyTorch's own ResNet34 model on CIFAR100 dataset [see assets](https://github.com/mhd53/resnet-from-torch/tree/master/_assets) for graph.
- Reached 73% validation accuracy on CIFAR10 dataset using ResNet34 model.
- Trained for approximately an hour and reached 75% validation accuracy on CIFAR10 dataset using ResNet50 model

## Author

* Website: morealfit
* Github: [@mhd53](https://github.com/mhd53)

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.

## 📃 Reference Papers:

- [Deep Residual Learning for Image Recognition, He et al. 2015](https://arxiv.org/pdf/1409.1556.pdf)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, He et al. 2015](https://arxiv.org/abs/1502.01852)
- [Very Deep Convolutional Networks For Large-Scale Image Recognition, Simonyan & Zisserman. 2015](https://arxiv.org/pdf/1409.1556.pdf)
- [Network In Network, Lin et al. 2014](https://arxiv.org/abs/1312.4400)

***
