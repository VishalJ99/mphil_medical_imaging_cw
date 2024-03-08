import os
import torch
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import dicom_dir_to_3d_arr, load_npz, set_device, seed_everything
from datasets import ImageSegmentationDataset
from models import UNet2D, SimpleUNet
from losses import SoftDiceBCECombinedLoss, SoftDiceLoss
from tqdm import tqdm
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt


quick_test = True
device = set_device()
img_dir = "Dataset/Images"
mask_dir = "Dataset/Segmentations"
seed_everything(42)
print("numpy random sequence:", np.random.rand(5))
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="medical_imaging_1",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "SimpleUNet",
        "dataset": "LCTSC",
        "epochs": 10,
    },
)

# Load the 3D volume and mask for each case.
all_slices_img_mask_case_dict = []
for case_name in os.listdir(img_dir):
    case_fpath = os.path.join(img_dir, case_name)
    mask_fpath = os.path.join(mask_dir, case_name + "_seg.npz")

    case_arr = dicom_dir_to_3d_arr(case_fpath, np.float32)
    mask_arr = load_npz(mask_fpath)

    # Create a tuple for each slice in the 3D volume.
    for slice in range(case_arr.shape[0]):
        case_tuple = (f"{case_name}_{slice}", case_arr[slice], mask_arr[slice])
        all_slices_img_mask_case_dict.append(case_tuple)

# Split 2D slice data into train and validation sets.
train, test = train_test_split(
    all_slices_img_mask_case_dict, test_size=0.33, random_state=42
)

train_case_ids, train_images, train_masks = list(zip(*train))
test_case_ids, test_images, test_masks = list(zip(*test))

if quick_test:
    # pick central slice since it normally contains a slice with a large
    # amount of the lungs.
    print("[INFO] Quick test mode enabled.")
    print("[INFO] Only using only one case for training and testing.")

    train_case_ids = train_case_ids[len(train_case_ids) // 2]
    train_images = [train_images[len(train_case_ids) // 2]]
    train_masks = [train_masks[len(train_case_ids) // 2]]

    test_case_ids = test_case_ids[len(test_case_ids) // 2]
    test_images = test_images[len(test_case_ids) // 2]
    test_masks = test_masks[len(test_case_ids) // 2]


# Define Train and Validation Transforms.
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Split the dataset into train and validation sets.
train_dataset = ImageSegmentationDataset(
    train_images,
    train_masks,
    train_case_ids,
    image_transform=train_transforms,
    mask_transform=train_transforms,
)

test_dataset = ImageSegmentationDataset(
    test_images,
    test_masks,
    test_case_ids,
    image_transform=test_transforms,
    mask_transform=test_transforms,
)

# Define the dataloaders.
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=3, shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=True)

# Define model, loss function, and optimizer.
# model = UNet2D(in_channels=1, out_channels=1).to(device)
model = SimpleUNet(in_channels=1, out_channels=1).to(device)
# state_dict = torch.load("SimpleUNet_v3.pt", map_location=device)
# model.load_state_dict(state_dict)

optim = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()
class_labels = {0: "background", 1: "lung"}
# Train the model.
for epoch in range(50):
    model.train()
    # Initialize variables to accumulate loss and track the number of batches.
    total_loss = 0
    num_batches = 0

    # Wrap your data loader with tqdm for a progress bar.
    progress_bar = tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}"
    )
    for i, (images, masks, _) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        optim.zero_grad()
        preds = model(images)

        loss = loss_fn(preds.to(torch.float32), masks.to(torch.float32))
        loss.backward()
        optim.step()
        wandb.log({"loss": loss.item()})
        # Update total loss and batch count.
        total_loss += loss.item()
        num_batches += 1

        # Visualise predictions vs ground truth
        fig, ax = plt.subplots(1, 3)
        img = images[0].squeeze().cpu().detach().numpy()
        mask = masks[0].squeeze().cpu().detach().numpy()
        pred = preds[0]
        pred = (torch.sigmoid(pred)).squeeze().cpu().detach().numpy()
        ax[0].imshow(mask)
        ax[0].set_title("Ground Truth")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        ax[2].imshow(img, cmap="gray")
        ax[2].set_title("Image")
        plt.show()

        # Update the progress bar with the latest loss.
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average loss over all batches.
    avg_loss = total_loss / num_batches

    summary_stats = f"Epoch: {epoch}, Avg. Loss: {avg_loss:.4f}"
    print(summary_stats)
    wandb.log(
        {
            "lung_ct_seg_test": wandb.Image(
                images.squeeze().cpu().detach().numpy(),
                masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": class_labels,
                    },
                },
            )
        }
    )


wandb.finish()


# # # Evaluate the model.
# # model.eval()
# # with torch.no_grad():
# #     for i, (images, masks, _) in enumerate(test_dataloader):
# #         images = images.to(device)
# #         masks = masks.to(device)
# #         preds = model(images)
# #         loss = loss_fn(preds, masks)
# # print(f"Validation Loss: {loss.item()}")
