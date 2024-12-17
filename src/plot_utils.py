import matplotlib.pyplot as plt


def plot_training_loss(losses, title="Training Loss Over Epochs", xlabel="Epoch", ylabel="Loss"):
    """
    Plot training loss over epochs.

    Args:
        losses (list): List of loss values for each epoch.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
