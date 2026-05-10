# Interactive Math Tutor (Digit Recognizer ML Project)

A fun and interactive web application that helps users practice basic addition by solving math problems and drawing their answers. The application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits in real-time.

## Features

- **Math Problem Generation:** Automatically generates simple addition problems.
- **Interactive Drawing Canvas:** Users can draw their numerical answers directly in the web browser.
- **Real-time Predictions:** Instantly predicts the drawn digit using a backend Flask API.
- **Gamified Feedback:** Displays success messages and confetti animations for correct answers.
- **Robust CNN Model:** Uses an improved CNN with data augmentation, batch normalization, and dropout to accurately classify digits.

## Project Structure

- `app.py` - Flask web application that serves the frontend and handles prediction requests.
- `templates/index.html` - The frontend HTML/CSS/JS for the interactive drawing canvas and game logic.
- `model training.ipynb` - Jupyter notebook containing the code used for training the CNN model with data augmentation.
- `digit_recognizer_improved110.h5` - The pre-trained Keras model used for inference in the Flask app.
- `Dockerfile` - For containerizing the application using Docker.
- `requirements.txt` - Python dependencies for running the project.

## Installation and Usage

You can run this project either using Docker (recommended) or locally with Python.

### Option 1: Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Build the Docker image:
   ```bash
   docker build -t math-tutor-app .
   ```

3. Run the container:
   ```bash
   docker run -p 5006:5006 math-tutor-app
   ```

4. Open your web browser and navigate to `http://localhost:5006`.

### Option 2: Running Locally

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5006`.

## Training the Model (Optional)

If you wish to explore how the model was trained or train it yourself:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open and run `model training.ipynb`.

## Dependencies

- **Flask** - Web framework for the backend API
- **TensorFlow / Keras** - Deep learning framework for loading the model and making inferences
- **OpenCV & Pillow** - Image processing and manipulation
- **NumPy** - Numerical computations

*(For training, additional libraries such as Matplotlib, Seaborn, and Scikit-learn are used.)*

## Model Architecture

The improved model used in this project includes:
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
