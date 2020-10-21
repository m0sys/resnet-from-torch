from torchvision import datasets, transforms
from PIL import Image
from base import BaseDataLoader

CIFAR_100_MEAN = [0.507, 0.487, 0.441]
CIFAR_100_STD = [0.267, 0.256, 0.276]


class Cifar100DataLoader(BaseDataLoader):
    """
    CIFAR100 dataloader with train/val split.
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        trsfm = _apply_cifar_trsfm(
            training, _create_cifar_normalization(CIFAR_100_MEAN, CIFAR_100_STD)
        )

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


def _create_cifar_normalization(mean, std):
    return transforms.Normalize(mean=mean, std=std)


def _apply_cifar_trsfm(training: bool, normalize: transforms.Normalize):
    if training:
        trsfm = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        trsfm = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return trsfm
