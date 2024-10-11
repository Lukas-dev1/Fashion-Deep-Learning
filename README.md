![Alt text](https://i.ibb.co/y5Hvd1Q/Figure-1.webp)


# FashionMNIST Deep Learning Project

This project uses deep learning to classify images from the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The dataset contains 28x28 grayscale images of various clothing items, and this model aims to classify them into one of 10 categories.

## Files in this Repository

- **`Fashion Deep learning.ipynb`**: Jupyter notebook that contains the training process for the deep learning model, including dataset preparation, model setup, and training loop.
- **`model_300.pth`**: The saved PyTorch model file after training. You can load this model for inference or further testing.
- **`model_testing.py`**: Python script for testing the trained model on new data or the test set.

## Setup Instructions

### Prerequisites

Make sure you have the following installed:

- Python 3.6+
- PyTorch
- torchvision
- tqdm
- matplotlib
- scikit-learn

You can install the dependencies using:

```bash
pip install torch torchvision tqdm matplotlib scikit-learn
```

### Dataset

The FashionMNIST dataset is automatically downloaded by the PyTorch `datasets` module in the notebook, so no manual setup is required for it.

## How to Use

### 1. Training the Model

To train the model, you can use the Jupyter notebook `Fashion Deep learning.ipynb`. It will load the dataset, define a neural network, and train it for image classification.

### 2. Testing the Model

If you'd like to test the model with pre-trained weights (`model_300.pth`), you can run the provided script `model_testing.py`. Make sure that the model file is in the same directory as the script.

```bash
python model_testing.py
```

The script will load the saved model and run predictions on the test dataset.

### 3. Evaluating Model Performance

The notebook also includes code for evaluating the model using accuracy and a classification report. You can check the classification metrics using `sklearn.metrics.classification_report`.

## Results

The trained model achieves reasonable accuracy on the test set, distinguishing between different types of clothing items such as shirts, shoes, and bags.

## Future Improvements

- Experiment with deeper architectures like ResNet or EfficientNet.
- Tune hyperparameters like learning rate, batch size, and epochs to improve performance.
- Apply data augmentation to enhance generalization.

## License

This project is licensed under the MIT License. Feel free to use and modify it for your own purposes.
