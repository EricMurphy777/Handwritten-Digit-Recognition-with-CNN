# Handwritten Digit Recognition with CNN (PyTorch)

This project implements a simple Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset.

## Motivation
The project was inspired by my coursework in Neuroscience and Brain-Inspired Intelligence.  
In particular, I was interested in how hierarchical visual processing in biological systems can be abstracted into computational models such as CNNs.

## Model Architecture
- Two convolutional layers for local feature extraction (receptive fields)
- Max pooling for spatial abstraction
- Fully connected layers for classification

## Implementation Details
- Framework: PyTorch (CPU-only)
- Dataset: MNIST
- Loss function: Cross-Entropy Loss
- Optimizer: Adam

During implementation, I encountered a dimension mismatch issue between convolutional outputs and the fully connected layer.  
By analyzing the feature map sizes, I corrected the network design and successfully trained the model.

## Key Takeaways
- Practical understanding of how spatial dimensions change through convolution and pooling
- Hands-on experience debugging neural network architectures
- A concrete connection between biological visual perception and artificial neural networks

## Future Work
- Visualize learned convolutional filters
- Compare CNN performance with fully connected networks

- Explore more biologically inspired architectures

