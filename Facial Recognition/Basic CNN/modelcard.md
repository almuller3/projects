# Model Card: Expanded CNN for Facial Recognition Using the LFW Dataset

## Model Details

### Overview
This model is an expanded Convolutional Neural Network (CNN) designed to perform facial recognition on the **Labeled Faces in the Wild (LFW)** dataset. It aims to classify facial images based on identity, using several convolutional layers to extract hierarchical features from images. The model is built using **PyTorch** and includes several layers of batch normalization and dropout to improve performance and reduce overfitting.

### Model Architecture
- **Input**: 3-channel (RGB) images, resized to 64x64 pixels.
- **Convolutional Layers**:
  - 4 convolutional layers with increasing filter sizes (32, 64, 128, 256).
  - Each layer is followed by batch normalization and ReLU activation.
  - Max-pooling is applied after each convolutional block to reduce spatial dimensions.
- **Fully Connected Layers**:
  - 2 fully connected (FC) layers for feature mapping (512 units and 128 units).
  - A final output layer with the number of units equal to the number of unique identities in the LFW dataset.
- **Dropout**: Applied after each FC layer to prevent overfitting.
- **Output**: A class label representing the identity of the face in the image.
  
### Loss Function
The model uses **CrossEntropyLoss**, which is suitable for multi-class classification problems where the task is to classify images into distinct identities.

### Optimizer
The model is trained using the **Adam optimizer** with a learning rate of `0.001`.

---

## Intended Use

### Primary Use Case
The model is intended for **facial recognition** tasks where the objective is to classify facial images into distinct identities. It is ideal for educational purposes, demonstration, and experimentation with convolutional neural networks and facial recognition techniques.

### Limitations
- **Dataset Size**: The LFW dataset contains relatively few images per identity, which may limit the model’s ability to generalize.
- **Image Quality**: LFW images are mostly low-resolution (250x250 pixels), which could limit the model’s accuracy in real-world, high-resolution settings.
- **Not Fine-Tuned**: The model does not use transfer learning or pre-trained models, which may result in lower accuracy compared to state-of-the-art models that use large-scale datasets and pre-training.

### Performance
The model is evaluated on the **training set** using accuracy as the primary metric. The current version does not include evaluation on a validation or test set, but such evaluation is recommended for deployment in real-world scenarios.

### Ethical Considerations
- **Bias**: The LFW dataset is biased toward certain demographics, especially in terms of gender and race. This model may not perform well for underrepresented groups.
- **Privacy**: Facial recognition models can raise privacy concerns, especially if used in surveillance systems. This model is designed for educational purposes, and any deployment should consider privacy and legal regulations.

---

## Training Data

### Dataset
- **Labeled Faces in the Wild (LFW)**: A public dataset containing images of faces labeled with identities.
- **Dataset Size**: 13,233 images of 5,749 individuals, with the majority having only one image.
- **Preprocessing**:
  - Images are resized to 64x64 pixels.
  - Each image is normalized using mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5].

---

## Training Procedure

### Environment
- **Framework**: PyTorch
- **Hardware**: The model is designed to run on both **CPU** and **GPU** (preferably using CUDA on Google Colab or similar environments).
  
### Hyperparameters
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Number of Epochs**: 10 (can be increased for better performance)
  
---

## Model Limitations

### Known Limitations
- **Overfitting**: Without regularization or augmentation, the model might overfit the training data, particularly with a small dataset like LFW.
- **Not State-of-the-Art**: The model is relatively simple and not fine-tuned for high-performance facial recognition. It's intended for educational purposes.
  
---

## Recommendations

### Usage Instructions
- **Training**: For better generalization, consider using data augmentation techniques (e.g., random cropping, flipping, or rotation).
- **Evaluation**: Split the dataset into training and validation sets for more accurate performance metrics. Consider using pre-trained models for improved accuracy.
- **Deployment**: This model is intended for educational use and should not be deployed in sensitive real-world applications without further testing and tuning.

---

## Authors
This model card was prepared by **Al Muller** and the model was developed using PyTorch in a teaching/learning context.

