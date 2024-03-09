import torch


class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self, images, masks, case_ids, image_transform=None, mask_transform=None
    ):
        self.images = images
        self.masks = masks
        self.case_ids = case_ids
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        case_id = self.case_ids[idx]

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return (image, mask, case_id)
