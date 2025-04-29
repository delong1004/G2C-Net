from torchvision import datasets, transforms
import torch
from torchsampler import ImbalancedDatasetSampler


def load_data(imbalance, data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.RandomResizedCrop([256, 256]),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.CenterCrop(256),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(imbalance, data, batch_size=batch_size,
                                  shuffle=True if train else False,
                                  num_workers=num_workers, **kwargs, drop_last=True if train else False, pin_memory=True)
    data_list = data.class_to_idx
    return data_loader, data_list


def get_data_loader(imbalance, dataset, batch_size, shuffle=True, num_workers=0, infinite_data_loader=False, **kwargs):
    if imbalance:
        return torch.utils.data.DataLoader(dataset, sampler=ImbalancedDatasetSampler(dataset), batch_size=batch_size, num_workers=num_workers, **kwargs)
    else:
        if not infinite_data_loader:
            return torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        else:
            return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0
