import sys
import os
from losses import SoftDiceLoss
import torch

sys.path.append(os.path.abspath("src"))


def test_soft_dice_loss():
    # Create known pred and target.
    mock_pred_1 = (
        torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]).float().view(1, 1, 3, 3)
    )

    mock_pred_2 = (
        torch.tensor([[[0, 0, 0], [0, 0.5, 0], [0, 0, 0]]]).float().view(1, 1, 3, 3)
    )

    mock_label = (
        torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float().view(1, 1, 3, 3)
    )

    # Create the loss function with no smoothing or sigmoid.
    loss_fn = SoftDiceLoss(eps=0, apply_sigmoid=False)

    # Compute the loss.
    loss_1 = loss_fn(mock_pred_1, mock_label)
    loss_2 = loss_fn(mock_pred_2, mock_label)

    # Test the loss.
    assert loss_1 - (1 - 0.5) < 1e-6, f"Expected 0.5, but got {loss_1}"
    assert loss_2 - (1 - 1 / 3.5) < 1e-6, f"Expected 0.286, but got {loss_2}"
