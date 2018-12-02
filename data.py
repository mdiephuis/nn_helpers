import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# todo: set or infer the output size


class Loader(object):
    def __init__(self, dataset_ident, file_path, download, shuffle, batch_size, data_transform, target_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        # set the dataset
        # NOTE: will need a refractor one we load more different datasets, that require custom classes
        loader_map = {
            'mnist': datasets.MNIST,
            'fashion': datasets.FashionMNIST
        }

        num_class = {
            'mnist': 10,
            'fashion': 10
        }

        # If download true and target dir doesn't exist, save into new dir named after the dataset
        # and append this new dir to loader file_path
        if download is True:
            file_path = os.path.join(file_path, dataset_ident)
            if not os.path.isdir(file_path):
                os.makedirs(file_path)

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       data_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        # infer and set size, idea from:
        # https://github.com/jramapuram/helpers/
        tmp_batch, _ = self.train_loader.__iter__().__next__()
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]
        self.batch_size = batch_size

    @staticmethod
    def get_dataset(dataset, file_path, download, data_transform, target_transform):

        # Check for transform to be None, a single item, or a list
        # None -> default to transform_list = [transforms.ToTensor()]
        # single item -> list
        if not data_transform:
            data_transform = [transforms.ToTensor()]
        elif not isinstance(data_transform, list):
            data_transform = list(data_transform)

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=transforms.Compose(data_transform),
                                target_transform=target_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=transforms.Compose(data_transform),
                               target_transform=target_transform)

        return train_dataset, test_dataset
