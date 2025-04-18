###Classifying Galaxies Using Convolutional Neural Networks

##This project tackles the challenge of classifying galaxies based on their visual morphology using a deep learning approach. Leveraging data from the Galaxy Zoo project — a crowd-sourced initiative where volunteers label millions of galaxy images — we trained a Convolutional Neural Network (CNN) to automatically classify deep-space galaxies into four distinct categories based on identifying visual characteristics.

##📌 Problem Statement
Astronomers face a major bottleneck when it comes to leveraging large-scale telescope imagery: unannotated data. This model aims to automate the classification process to support scientific discovery and reduce manual annotation effort.

##📁 Dataset
Source: Galaxy Zoo Project
Classes:
[1, 0, 0, 0]: Galaxies with no identifying characteristics
[0, 1, 0, 0]: Spiral galaxies
[0, 0, 1, 0]: Elliptical galaxies
[0, 0, 0, 1]: Irregular/odd galaxies
🧠 Model
Architecture: Convolutional Neural Network (CNN) with multiple convolutional + max-pooling layers, followed by dense layers with softmax output
Libraries: TensorFlow, Keras, NumPy, Matplotlib
Techniques Used:
Data preprocessing & normalization
One-hot encoding of labels
Model checkpointing and early stopping
Training-validation split
Evaluation using accuracy and confusion matrix
📊 Results
Achieved high validation accuracy in galaxy classification
Visualized performance metrics and misclassifications
Demonstrated the ability to detect “odd” or uniquely structured galaxies
🚀 Future Improvements
Explore transfer learning with pre-trained models like ResNet
Increase dataset size or add data augmentation
Deploy model as a web application for real-time galaxy classification
