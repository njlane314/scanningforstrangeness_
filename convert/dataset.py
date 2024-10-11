# data.py

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()

    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

class SegmentationDataset(Dataset):
    """Dataset suitable for segmentation tasks.
    """

    def __init__(self, image_dir, mask_dir, filenames, transform=False, device=torch.device('cuda:0')):
        """Constructor.

            Args:
                image_dir: The directory containing the images
                mask_dir: The directory containing the masks
                filenames: The filanems for the images associate with this dataset
                transform: Whether or not to transform the items (default: False).
                device: The device on which tensors should be created (default: 'cuda:0')
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = filenames
        self.device = device


    def __len__(self):
        """Retrieve the number of samples in the dataset.

            Returns:
                The number of samples in the dataset
        """
        return len(self.filenames)


    def __getitem__(self, idx):
        """Retrieve a sample from the dataset.

            Args:
                idx: The index of the sample to be retrieved

            Returns:
                The sample requested
        """
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        with open(img_name, 'rb') as file:
            image = np.load(file)['arr_0']

        mask_name = os.path.join(self.mask_dir, self.filenames[idx])
        with open(mask_name, 'rb') as file:
            mask = np.load(file)['arr_0']

        image = torch.as_tensor(np.expand_dims(image, axis=0), device=self.device, dtype=torch.float)
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.long)

        if self.transform:
            should_hflip = True if torch.rand(1) > 0.5 else False
            should_vflip = True if torch.rand(1) > 0.5 else False
            should_transpose = True if torch.rand(1) > 0.5 else False
            # need to check that these make sense in the context of pixel classification
            if should_hflip:
                image = tv.transforms.functional.hflip(image)
                mask = tv.transforms.functional.hflip(mask)
            if should_vflip:
                image = tv.transforms.functional.vflip(image)
                mask = tv.transforms.functional.vflip(mask)
            if should_transpose:
                image = image.transpose(1, 2)
                mask = mask.transpose(1, 2)

        return (image, mask)


class SegmentationBunch():
    """Associates batches of training, validation and testing datasets suitable
        for segmentation tasks.
    """

    def __init__(self, root_dir, image_dir, mask_dir, batch_size, train_pct=None, valid_pct=0.1,
                 test_pct=0.0, transform=False, device=torch.device('cuda:0')):
        """Constructor.

            Args:
                root_dir: The top-level directory containing the images
                image_dir: The relative directory containing the images
                mask_dir: The relative directory containing the masks
                batch_size: The batch size
                valid_pct: The fraction of images to be used for validation (default: 0.1)
                test_pct: The fraction of images to be used for testing (default: 0.0)
                transform: Whether or not to transform the items (default: False)
                device: The device on which tensors should be created (default: 'cuda:0')
        """
        assert((valid_pct + test_pct) < 1.)
        image_dir = os.path.join(root_dir, image_dir)
        mask_dir = os.path.join(root_dir, mask_dir)
        image_filenames = np.array(next(os.walk(image_dir))[2])
        print(image_filenames)
        n_files = len(image_filenames)
        valid_size = int(n_files * valid_pct)
        train_size = n_files - valid_size if train_pct is None else int(n_files * train_pct)

        sample = np.random.permutation(n_files)
        train_sample = sample[valid_size:] if not train_size else \
            sample[valid_size:valid_size + train_size]
        valid_sample = sample[:valid_size]

        train_ds = SegmentationDataset(image_dir, mask_dir, image_filenames[train_sample], transform, device)
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

        valid_ds = SegmentationDataset(image_dir, mask_dir, image_filenames[valid_sample], None, device)
        self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)


    def count_classes(self, num_classes):
        """Count the number of instances of each class in the training set

            Args:
                num_classes: The number of classes in the training set

            Returns:
                A list of the number of instances of each class
        """
        count = np.zeros(num_classes)
        print("Getting class counts...", flush=True)

        for batch in tqdm(self.train_dl, desc="Counting Classes"):
            _, truth = batch
            unique = torch.unique(truth)
            counts = torch.stack([(truth == x_u).sum() for x_u in unique])
            unique = [ u.item() for u in unique ]
            counts = [ c.item() for c in counts ]
            this_dict = dict(zip(unique, counts))
            for key in this_dict:
                count[key] += this_dict[key]

        return count
