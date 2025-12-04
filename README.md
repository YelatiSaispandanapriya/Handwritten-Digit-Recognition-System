# Handwritten Digit Recognition System.

This project implements a **Handwritten Digit Recognition System** trained on the classic **MNIST dataset** using a **manually built deep neural network from scratch** — without using deep-learning libraries like TensorFlow or PyTorch.

The entire neural network — including **weight initialization, forward propagation, ReLU activation, softmax output, cross-entropy loss, backpropagation, mini-batch gradient descent**, and **evaluation metrics** — is implemented using **NumPy only**.

## Project Highlights

- Classifies handwritten digits **0–9**
- Fully manual neural network built using NumPy
- Implements forward/backprop, softmax, ReLU, loss, etc.
- Achieved **96.89% test accuracy**
- Includes confusion matrix and classification report
- Loss curve visualization included

## Model Architecture
- Input: 784
- Hidden 1: 128 (ReLU)
- Hidden 2: 64 (ReLU)
- Output: 10 (Softmax)

## Results
- Accuracy: **0.9689**
- Macro Precision: 0.9688
- Macro F1-score: 0.9687

## Project Structure
```
README.md
neural_network_mnist.py
loss_curve.png (optional)
model_params.pkl (optional)
```

## How to Run
```
pip install numpy matplotlib scikit-learn
python neural_network_mnist.py
```

## Future Improvements
- Add Adam optimizer
- Add dropout
- Deploy using Flask/Streamlit
- GPU acceleration with CuPy
