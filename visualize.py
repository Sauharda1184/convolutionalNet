import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def visualize_activations(model, data_iterator):
    """
    Visualize activations of the convolutional layers in the model.
    
    Args:
        model: A TensorFlow Keras model
        data_iterator: An iterator providing batches of data
    """
    # Get one batch of data
    try:
        images, _ = next(iter(data_iterator))
    except:
        # If the iterator doesn't support unpacking like this
        images = next(iter(data_iterator))[0]
    
    # Make sure we have at least one image
    if len(images) == 0:
        print("No images available for visualization")
        return
    
    # Take the first image for visualization
    image = images[0:1]
    
    # We need to create a new functional model that connects to the convolutional layers
    # First, identify the convolutional layers in the model
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name:
            conv_layers.append((i, layer))
    
    if not conv_layers:
        print("No convolutional layers found in the model")
        return
    
    # Create a list to store visualizations
    activations = []
    
    # Create a new sequential model up to each conv layer and get output
    for i, (layer_idx, _) in enumerate(conv_layers):
        temp_model = tf.keras.models.Sequential()
        
        # Add input layer with same shape as our model's input
        temp_model.add(tf.keras.layers.InputLayer(input_shape=(128, 128, 3)))
        
        # Add all layers up to and including this conv layer
        for j in range(layer_idx + 1):
            temp_model.add(model.layers[j])
        
        # Get the output from this partial model
        activation = temp_model.predict(image)
        activations.append(activation)
    
    # Plot the activations
    plt.figure(figsize=(15, 5))
    layer_names = [f"Conv Layer {i+1}" for i in range(len(conv_layers))]
    
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        # Number of features in the feature map
        n_features = layer_activation.shape[-1]
        n_to_display = min(8, n_features)
        
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
        
        # Create a grid of subplots
        n_cols = n_to_display
        display_grid = np.zeros((size, size * n_cols))
        
        # Fill the grid with the feature maps
        for col in range(n_cols):
            feature = layer_activation[0, :, :, col]
            feature -= feature.mean()
            feature /= feature.std() + 1e-8
            feature = feature * 64 + 128
            feature = np.clip(feature, 0, 255).astype('uint8')
            display_grid[:, col * size : (col + 1) * size] = feature
        
        # Plot the grid
        plt.subplot(1, len(activations), i + 1)
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.tight_layout()
    plt.savefig('activations.png')
    print("Activation visualization saved as 'activations.png'")
    plt.close() 