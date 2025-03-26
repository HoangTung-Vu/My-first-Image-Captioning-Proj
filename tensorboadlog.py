import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_logs(log_dir):
    """Loads tensorboard logs from the given directory."""
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError("No TensorBoard logs found in the specified directory.")
    
    # Load the latest event file
    event_file = sorted(event_files, key=os.path.getmtime)[-1]
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    return event_acc

def extract_scalars(event_acc, tag):
    """Extracts scalar values from a given tag in the event accumulator."""
    if tag not in event_acc.Tags()["scalars"]:
        print(f"Tag {tag} not found in logs.")
        return [], []
    
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_metrics(log_dir):
    """Plots batch-wise training loss and epoch-wise training/validation loss."""
    event_acc = load_tensorboard_logs(log_dir)

    # Extract training batch loss
    batch_steps, batch_loss = extract_scalars(event_acc, "Training/BatchLoss")

    # Extract training and validation epoch loss
    train_steps, train_loss = extract_scalars(event_acc, "Training/EpochLoss")
    val_steps, val_loss = extract_scalars(event_acc, "Validation/Loss")

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot batch loss
    axes[0].plot(batch_steps, batch_loss, label="Batch Loss", color="blue", alpha=0.7)
    axes[0].set_xlabel("Batch Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Batch Loss")
    axes[0].legend()
    axes[0].grid()

    # Plot epoch loss
    axes[1].plot(train_steps, train_loss, label="Training Loss", marker="o", color="green")
    axes[1].plot(val_steps, val_loss, label="Validation Loss", marker="s", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

# Set your TensorBoard log directory
log_directory = "runs/image_captioning"

# Run visualization
plot_metrics(log_directory)
