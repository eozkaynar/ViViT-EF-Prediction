import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_npy_files(data_dir, output_dir, params):
    """
    Find, load, plot, and save .npy files based on parameterized filenames.

    Args:
        data_dir (str): Directory containing .npy files.
        output_dir (str): Directory to save the plots.
        params (dict): Dictionary with parameters to match filenames.

    Example:
        params = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "batch_size": 20,
            "num_heads": 8,
            "num_layers": 8,
            "projection_dim": 128
        }
    """
    # Generate the filename pattern
    filename_pattern = (
        f"lr_{params['lr']}_wd_{params['weight_decay']}_bs_{params['batch_size']}_nh_{params['num_heads']}_nl_{params['num_layers']}_pd_{params['projection_dim']}"
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the parameterized directory
    param_dir = os.path.join(data_dir, filename_pattern)

    if not os.path.exists(param_dir):
        print(f"Directory not found: {param_dir}")
        return

    # Load the .npy files
    train_losses_path = os.path.join(param_dir, "train_losses.npy")
    val_losses_path = os.path.join(param_dir, "val_losses.npy")

    if not os.path.exists(train_losses_path) or not os.path.exists(val_losses_path):
        print("Required .npy files are missing in the directory.")
        return

    train_losses = np.load(train_losses_path)
    val_losses = np.load(val_losses_path)

    # Ensure arrays are not empty
    if train_losses.size == 0 or val_losses.size == 0:
        print(f"One of the .npy files is empty: {param_dir}")
        return

    # Generate iterations
    iterations = np.linspace(0, 10, num=len(train_losses))  # DIKKAT DIKKAT
    iterations1 = np.linspace(0, 10, num=len(val_losses))  # DIKKAT DIKKAT

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label="Training Loss")
    plt.plot(iterations1, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the plot
    output_file = os.path.join(output_dir, f"{filename_pattern}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot: {output_file}")

# Example usage
data_dir = "/home/eda/Desktop/Video-Vision-Transformer/hyperparam_outputs"
output_dir = "/home/eda/Desktop/Video-Vision-Transformer/hyperparam_outputs/images"
params = {
    "lr": 0.0001,
    "weight_decay": 1e-5,
    "batch_size": 16,
    "num_heads": 8,
    "num_layers": 8,
    "projection_dim": 1024
}

plot_and_save_npy_files(data_dir, output_dir, params)
