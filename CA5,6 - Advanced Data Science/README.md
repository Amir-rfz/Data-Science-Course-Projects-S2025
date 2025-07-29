# Project 4: Deep Learning Applications

## Project Overview

This repository contains the solutions for a three-part assignment focused on applying core deep learning architectures to solve a variety of real-world problems. The project is divided into three distinct tasks:

1.  **Tabular Data Prediction:** Using a Multi-Layer Perceptron (MLP) to predict football match outcomes.

2.  **Image Classification:** Building a Convolutional Neural Network (CNN) to classify images of flowers.

3.  **Time-Series Forecasting:** Implementing a Recurrent Neural Network (RNN) to predict future Bitcoin prices.

Each task is self-contained and demonstrates the end-to-end process of data preparation, model design, training, and evaluation using the PyTorch framework.

### Task 1: MLP for FIFA World Cup Prediction

#### Objective

To build and train a Multi-Layer Perceptron (MLP) model to predict the outcome of international football matches. The model is trained on historical data from World Cup qualifiers and then used to simulate the entire FIFA World Cup 2022 tournament, from the group stage to the final.

#### Methodology

1.  **Data Preparation:** The dataset of international matches was loaded. Features were selected to represent the statistical differences between the two competing teams (e.g., historical wins, goals scored, etc.), while team names were excluded to prevent the model from learning biases. The target variable (`status`: win, tie, loss) was label-encoded.

2.  **Preprocessing:** The data was split into training and testing sets. Numerical features were standardized using `StandardScaler`, which was fit *only* on the training data to prevent data leakage.

3.  **MLP Architecture:** A flexible MLP was designed using PyTorch's `nn.Module`. The architecture consists of several fully connected (linear) layers with ReLU activation functions and dropout for regularization, outputting logits for the three possible outcomes.

4.  **Training & Evaluation:** The model was trained using the Cross-Entropy Loss criterion and the Adam optimizer. The training loop iterated for a set number of epochs, updating the model's weights to minimize the loss. The final model's performance was evaluated on the test set using accuracy.

5.  **World Cup Simulation:** The trained model was used to predict the outcome of each match in the FIFA World Cup 2022 group stage. The group standings were updated, and the top two teams from each group advanced to a simulated knockout stage until a final winner was predicted.

### Task 2: CNN for Flower Image Classification

#### Objective

To build, train, and compare two Convolutional Neural Network (CNN) models for classifying flower images from a multi-class dataset: a VGG-style network built from scratch and a fine-tuned pre-trained ResNet model.

#### Methodology

1.  **Dataset & Preprocessing:** The flower image dataset was loaded and split into training, validation, and test sets. All images were resized to 224x224 pixels and normalized.

2.  **Data Augmentation:** To improve model generalization, random transformations (rotations, horizontal flips, and color jitter) were applied to the training dataset only.

3.  **VGG-Style CNN from Scratch:** A custom CNN inspired by the VGG architecture was implemented. It features multiple blocks of stacked 3x3 convolutional layers followed by max-pooling layers, and a final classifier head with fully connected layers.

4.  **Fine-Tuning Pre-trained ResNet:** A pre-trained ResNet50 model was used. Fine-tuning was performed in stages:

    * First, only the final classification layer (the "head") was trained while the convolutional base was frozen.

    * Next, the last few layers of the convolutional base were unfrozen and trained with a low learning rate.

    * Finally, the entire network was trained with a very low learning rate.

5.  **Evaluation:** Both models were evaluated on the test set. Performance was compared using **Accuracy, Precision, Recall, F1-Score,** and the **Area Under the ROC Curve (AUC)**. Confusion matrices were also plotted for detailed error analysis.

### Task 3: RNN for Bitcoin Price Forecasting

#### Objective

To develop a Recurrent Neural Network (RNN) to predict future Bitcoin prices based on historical OHLCV (Open, High, Low, Close, Volume) data. The project also explores using a Long Short-Term Memory (LSTM) network for comparison.

#### Methodology

1.  **Data Exploration & Feature Engineering:** Historical Bitcoin data was loaded and analyzed. A custom target variable indicating potential profit/loss was engineered from the OHLCV features.

2.  **Sequence Creation:** The time-series data was transformed into input sequences and corresponding targets. A lookback window (e.g., 60 days) was used to create sequences of historical data to predict the target for the next time step.

3.  **RNN/LSTM Architecture:** An RNN model was built with recurrent layers followed by fully connected layers to produce a single regression output. An optional, more advanced LSTM-based model was also implemented to better capture long-term dependencies. Dropout was used for regularization.

4.  **Training & Evaluation:** The models were trained to minimize a regression loss function like Mean Squared Error (MSE). The performance was evaluated on a held-out test set using several metrics:

    * **MSE, RMSE, MAE:** To measure the magnitude of the prediction error.

    * **Mean Absolute Percentage Error (MAPE):** To express the error as a percentage.

    * **Cumulative Error (CE):** To assess the model's overall prediction bias.

5.  **Visualization:** The predicted values were plotted against the actual values over time to visually assess the model's ability to capture trends and turning points.

### How to Run the Code

Each task is contained within its own dedicated Jupyter Notebook. To run the code and reproduce the results, please follow these steps:

1.  **Install Dependencies:**

    First, ensure you have a Python environment set up. It is highly recommended to use a virtual environment. Install all the required packages using the `requirements.txt` file provided with this project:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebooks:**
    For each task, open the corresponding Jupyter Notebook (e.g., `World-cup.ipynb`, `flower_classification.ipynb`, `Final_RNN_DS.ipynb`).

3.  **Execute Cells:**
    Run the cells in each notebook sequentially from top to bottom. The notebooks are designed to be self-contained and will handle data loading, preprocessing, model training, and evaluation for that specific task.

### Technologies Used

* **Framework:** PyTorch

* **Core Libraries:** Pandas, NumPy, Scikit-learn

* **Visualization:** Matplotlib, Seaborn

* **Environment:** Jupyter Notebook, Google Colab (for GPU access)
