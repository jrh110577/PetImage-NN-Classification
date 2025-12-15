# PetImage-NN-Classification

This project implements a convolutional neural network (CNN) to classify images as either cats or dogs. Using Python and TensorFlow, the model learns visual patterns from a large image dataset and predicts the correct class for unseen images. The project demonstrates the full deep learning pipeline, from data preprocessing to model evaluation.

Architecture: 2-layer Convolutional Neural Network (CNN)

Frameworks: TensorFlow, NumPy

Task: Binary image classification (Cat vs Dog)

The CNN extracts spatial features from images through convolutional layers and uses fully connected layers for classification.

Dataset Size: 25,000+ labeled images (12,500 Dogs and 12,500 Cats)

Classes: Cats, Dogs

Preprocessing Steps:

Image resizing and normalization

Data augmentation (rotation, flipping, zooming)

Train / validation split to prevent overfitting

Training:

Hyperparameters tuned (learning rate, batch size, epochs)

Optimization using backpropagation and gradient descent

Evaluation Metrics:

Classification accuracy

Confusion matrix

Validation accuracy curves

Final Accuracy: ~83% on the validation set

The model demonstrates strong performance in distinguishing between cats and dogs, highlighting effective feature learning and model tuning.
