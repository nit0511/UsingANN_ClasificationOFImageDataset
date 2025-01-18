# Image Classification with ANN using MNIST Dataset

This repository contains a Python implementation of an Artificial Neural Network (ANN) to classify handwritten digits using the MNIST dataset. The project is built using TensorFlow and Keras and demonstrates model training, evaluation, and prediction workflows.

## Project Structure

- **`image_classification.ipynb`**: The main Jupyter Notebook containing the complete code for training and evaluating the ANN model.
- **`SAVED_MODELS/`**: Directory where trained models are saved.
- **`README.md`**: Documentation for understanding and running the project.

---

## Dependencies

The following libraries are required to run the code:

- `numpy`
- `matplotlib`
- `pandas`
- `os`
- `tensorflow`
- `seaborn`
- `time`

You can install these libraries using the following command:

```bash
pip install numpy matplotlib pandas tensorflow seaborn
```

---

## Steps to Run the Project

### 1. Load and Preprocess the Data

The MNIST dataset is loaded using TensorFlow's `keras.datasets` module. The dataset is split into training, validation, and test sets. The images are normalized to scale the pixel values between 0 and 1.

### 2. Define the Model

An ANN is constructed using TensorFlow's `Sequential` API:
- Input layer: Flattens the 28x28 image.
- Two hidden layers: Dense layers with 300 and 100 neurons, activated by ReLU.
- Output layer: Dense layer with 10 neurons, activated by softmax for classification.

### 3. Compile the Model

The model is compiled using the following:
- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `ADAM`
- Metrics: `accuracy`

### 4. Train the Model

The model is trained for 50 epochs using the training data, with validation data provided for performance monitoring.

### 5. Save the Model

A function `saveModel_path` is used to save the trained model in the `SAVED_MODELS/` directory. The filename includes a timestamp for uniqueness.

### 6. Evaluate the Model

The model's performance is evaluated on the test set, and metrics such as accuracy are displayed.

### 7. Visualize Training Metrics

Training history is plotted to visualize the loss and accuracy trends over epochs.

### 8. Make Predictions

The trained model predicts the first three test samples, and the images and predicted classes are displayed.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/image_classification_ann.git
   cd image_classification_ann
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook image_classification.ipynb
   ```

3. Run all cells to train the model, save it, and evaluate it.

4. Check the `SAVED_MODELS/` directory for the saved model file.

5. Load the saved model and use it for inference:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model("SAVED_MODELS/<model_filename>.h5")
   predictions = model.predict(new_data)
   ```

---

## Outputs

- **Model Summary**: Architecture of the ANN displayed in the console.
- **Training Metrics**: Plots showing the loss and accuracy trends.
- **Saved Model**: Trained model stored in the `SAVED_MODELS/` directory.
- **Test Accuracy**: Evaluated performance on the test set.
- **Predictions**: Visualizations of sample predictions.

---


## Acknowledgments

- **Dataset**: MNIST, provided by TensorFlow.
- **Libraries**: TensorFlow, Keras, Matplotlib, Numpy, and Pandas.

Feel free to contribute to the project by creating pull requests or reporting issues. Enjoy learning and building with TensorFlow!

