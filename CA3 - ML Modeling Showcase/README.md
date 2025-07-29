# Project 3: Machine Learning Modeling Showcase

## Project Overview

This repository contains the solutions for three distinct machine learning challenges, each designed as a practical, hands-on project and hosted as a Kaggle competition. The project covers three core areas of machine learning: binary classification, regression, and recommendation systems. The primary focus is on the end-to-end machine learning workflow, from data exploration and feature engineering to model training, evaluation, and prediction.

**Constraint:** Per the assignment guidelines, all solutions are implemented using traditional machine learning algorithms (e.g., Scikit-learn, XGBoost, LightGBM). Deep learning models are strictly prohibited.

---

### Task 1: Cancer Patient Survival Prediction (Classification)

#### Objective

To develop a binary classification model that accurately predicts the survival outcome for cancer patients. The model uses a rich dataset containing patient demographics, diagnostic information, and treatment histories to classify the `Survival_Status` as 'Alive' (1) or 'Deceased' (0).

#### Methodology

1.  **Data Exploration & Preprocessing:** The initial phase involved a thorough analysis of the dataset. Key preprocessing steps included:

    * Handling missing values in features like `Weight` and `Surgery_Date`.

    * Converting date columns (`Birth_Date`, `Diagnosis_Date`) into more useful features, such as `Age_at_Diagnosis`.

    * Applying one-hot encoding to categorical features like `Cancer_Type`, `Occupation`, and `Insurance_Type`.

    * Scaling numerical features to ensure they are on a comparable scale.

2.  **Feature Engineering:** New features were created to capture more predictive signals. For example, the duration between diagnosis and surgery was calculated as a potential indicator of outcome.

3.  **Model Selection & Training:** Several classification algorithms were evaluated, including Logistic Regression, Random Forest, and Gradient Boosting models (XGBoost, LightGBM). Models were trained and fine-tuned using cross-validation to prevent overfitting.

4.  **Evaluation:** The primary metric for the Kaggle competition is **Accuracy**. However, during development, a comprehensive evaluation was performed using **Precision, Recall, and F1-Score** to understand the model's performance in correctly identifying each class.

* **Kaggle Competition Link:** [Task 1 - Classification](https://www.kaggle.com/competitions/ds-ca-3-q-1/leaderboard) (1'st rank)

---

### Task 2: Daily Bike Rental Prediction (Regression)

#### Objective

To build a regression model that predicts the total number of daily bike rentals (`total_users`). The prediction is based on a dataset containing seasonal information, weather conditions, and calendar details.

#### Methodology

1.  **Data Exploration & Preprocessing:** The dataset was analyzed to understand the relationships between features and bike rental counts.

    * Date features were expanded to include day of the week, month, and year as distinct features.

    * Categorical features like `weather_condition` were encoded.

    * Numerical features such as `temperature`, `humidity`, and `wind_speed` were scaled.

2.  **Feature Selection:** To improve model performance and reduce noise, feature selection techniques were applied. This included analyzing feature correlation with the target variable and using statistical tests (like p-values) to identify the most significant predictors.

3.  **Model Selection & Training:** A variety of regression algorithms were tested, such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting Regressors. Hyperparameter tuning was performed to optimize the best-performing model.

4.  **Evaluation:** The Kaggle competition is ranked based on **Mean Squared Error (MSE)**. For a more complete analysis during development, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-Squared ($R^2$)** were also calculated to assess model fit and error magnitude.

* **Kaggle Competition Link:** [Task 2 - Regression](https://www.kaggle.com/competitions/ds-ca-3-q-2) (12'st rank)

---

### Task 3: Movie Rating Prediction (Recommender System)

#### Objective

To design and build a collaborative filtering recommendation system that predicts the rating a user would give to a movie they have not yet seen.

#### Methodology

1.  **Data Exploration:** The project utilized two datasets: one with user-movie ratings and another with user-user trust relationships. The data was explored to understand rating distributions and the structure of the user trust network.

2.  **Model Development:** A collaborative filtering approach was implemented. This could involve techniques like:

    * **Matrix Factorization:** Using algorithms like Singular Value Decomposition (SVD) to decompose the user-item interaction matrix into latent factors for users and items.

    * **Neighborhood-based Methods:** Calculating user-user or item-item similarity to make predictions.
    * The user trust data could be incorporated to weight the influence of "trusted" users more heavily in the predictions.

3.  **Training and Prediction:** The model was trained on the `train_data_movie_rate.csv` dataset. It was then used to predict the ratings for the user-item pairs listed in the `test_data.csv` file.

4.  **Evaluation:** The model's performance was evaluated using standard recommendation system metrics like **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

* **Kaggle Competition Link:** [Task 3 - Recommendation](https://www.kaggle.com/competitions/ds-ca-3-q-3) (6'st rank)

---

### Technologies Used

* **Language:** Python

* **Core Libraries:** Pandas, NumPy, Scikit-learn

* **Advanced ML Models:** XGBoost, LightGBM, CatBoost

* **Visualization:** Matplotlib, Seaborn

* **Environment:** Jupyter Notebook

### How to Run the Code

Each task is contained within its own dedicated Jupyter Notebook.

1.  **Clone the repository and navigate to the project folder.**

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Open the Jupyter Notebook** for the desired task (e.g., `Task1.ipynb`).

4.  **Run the cells sequentially.** The notebooks are structured to cover data loading, preprocessing, model training, evaluation, and finally, generating the `submission.csv` file in the format required by Kaggle.
