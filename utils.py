import numpy as np

def load_galaxy_data():
    """
    Function to load galaxy image data.
    In a real scenario, this would load from files or a dataset.
    This is a placeholder implementation.
    
    Returns:
        tuple: (input_data, labels) where input_data is a numpy array of shape (n_samples, 128, 128, 3)
              and labels is a one-hot encoded numpy array of shape (n_samples, 4)
    """
    # Create dummy data for demonstration
    # 100 samples of 128x128 RGB images
    n_samples = 100
    input_data = np.random.rand(n_samples, 128, 128, 3)
    
    # Create 4 classes with roughly equal distribution
    class_labels = np.random.randint(0, 4, size=n_samples)
    
    # One-hot encode the labels
    labels = np.zeros((n_samples, 4))
    for i, label in enumerate(class_labels):
        labels[i, label] = 1
    
    return input_data, labels 