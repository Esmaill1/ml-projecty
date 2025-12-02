# Digit Recognizer ML Project

A machine learning project for handwritten digit recognition using a CNN trained on the MNIST dataset.

## Project Structure

- **GUI streamlit.py** - Interactive Streamlit web app for drawing digits and getting real-time predictions
- **model training.ipynb** - Jupyter notebook for training an improved CNN model with data augmentation
- **not the model .ipynb** - Jupyter notebook with a simpler baseline model
- **digit_recognizer_improved110.h5** - Pre-trained Keras model for inference

## Features

- **Interactive Drawing Canvas** - Draw digits in a web interface
- **Real-time Predictions** - Get instant digit predictions as you draw
- **Improved Model** - CNN with batch normalization, dropout, and data augmentation
- **MNIST Dataset** - Trained on 60,000 training samples

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-projecty
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Web App

Launch the Streamlit app:
```bash
streamlit run "GUI streamlit.py"
```

The app will open in your browser at `http://localhost:8501`

### Train the Model

Open and run either Jupyter notebook:
```bash
jupyter notebook "model training.ipynb"
```

## Dependencies

### App Dependencies
- **streamlit** - Web framework for the GUI
- **streamlit-drawable-canvas** - Drawing component
- **tensorflow** - Deep learning framework for model inference
- **opencv-python** - Image processing
- **numpy** - Numerical computations
- **Pillow** - Image manipulation

### Training Dependencies
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **scikit-learn** - Machine learning utilities

## Model Architecture

The improved model includes:
- 2 convolutional blocks (32→64 filters)
- Batch normalization layers
- Max pooling
- Dropout for regularization
- Dense layers with 256 neurons
- Softmax output for 10-digit classification

## Model Performance

The model achieves high accuracy on the MNIST test set through:
- Data augmentation (rotation, zoom, shift)
- Batch normalization
- Dropout regularization
- Early stopping and learning rate reduction

## Author

Esmaill1
