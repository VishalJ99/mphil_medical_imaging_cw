import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import seed_everything


def main(metrics_csv_path):
    # Set random seed for reproducability.
    seed_everything(42)

    # Sort metrics csv by dice.
    df = pd.read_csv(metrics_csv_path)
    df = df.sort_values(by="dice")

    print("[INFO] Summary statistics table for Dice and Accuracy metrics:")
    print(
        "[NOTE] Dice count may be less than accuracy count due to NaN values\
        of background slices."
    )
    summary_statistics = df[["dice", "accuracy"]].describe()
    print(summary_statistics, end="\n\n")

    # Replace 'NaN' strings with numpy.nan
    df.replace("NaN", np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Log the slices with the best 3 dice scores.
    best_3 = df[-4:-1]
    print("[INFO] Best 3 slices (sorted by dice):")
    print(best_3.to_string(index=False), end="\n\n")

    # Log the slices with the worst 3 dice scores.
    worst_3 = df[:3]
    print("[INFO] Worst 3 slices (sorted by dice):")
    print(worst_3.to_string(index=False), end="\n\n")

    # Log the slices with dice scores in the [.25,.75] quartile interval.
    dice = df["dice"]
    dice_uq = np.percentile(dice, 75)
    dice_lq = np.percentile(dice, 25)

    lower_dice_mask = dice > dice_lq
    upper_dice_mask = dice < dice_uq

    in_between_dice_mask = lower_dice_mask * upper_dice_mask
    in_between_dice = dice[in_between_dice_mask]

    # Randomly shuffle the series.
    in_between_dice = in_between_dice.sample(frac=1)

    print("[INFO] Typical representative 3 slices (sorted by dice):")
    print(df.loc[in_between_dice[:3].index].to_string(index=False), end="\n\n")

    # Visualise metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Dice Scores")
    plt.boxplot(dice, flierprops=dict(marker="x", markerfacecolor="g", markersize=5))

    plt.subplot(1, 2, 2)
    plt.title("Accuracy Scores")
    accuracy = df["accuracy"]
    plt.boxplot(
        accuracy, flierprops=dict(marker="x", markerfacecolor="g", markersize=5)
    )

    plt.suptitle("Metric Boxplots")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument("metrics_csv", type=str, help="Path to the metrics csv file.")
    args = parser.parse_args()
    main(args.metrics_csv)
