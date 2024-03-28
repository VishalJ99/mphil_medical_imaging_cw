import numpy as np


class WinsoriseTransform:
    """
    Apply winsorisation to the image.
    """
    def __init__(self, limits=(1, 99)):
        self.limits = limits

    def __call__(self, img: np.ndarray) -> np.ndarray:
        winsorised_img = np.clip(
            img, np.percentile(img, self.limits[0]), np.percentile(img, self.limits[1])
        )
        return winsorised_img


class NormaliseTransform:
    """
    Normalise the image to the range [new_min, new_max].
    """
    def __init__(self, new_min=0, new_max=1):
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, img: np.ndarray) -> np.ndarray:
        min_val = np.min(img)
        max_val = np.max(img)
        img_normalised = (img - min_val) / (max_val - min_val)
        img_normalised = img_normalised * (self.new_max - self.new_min) + self.new_min
        return img_normalised
