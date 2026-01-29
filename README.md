# High-Precision Weight Prediction via Physics-Informed Ensemble Learning

## 📌 Project Overview
This project involves building a machine learning model to predict individual body weight using data from the **National Health Interview Survey (NHIS) 2020**.

Unlike standard regression approaches, this solution utilizes a **Physics-Informed Machine Learning** strategy. By integrating domain knowledge (the biological relationship between BMI, height, and mass) directly into the feature engineering pipeline, the model achieves a competition-grade **Mean Squared Error (MSE) of ~182.0**, with a generalization gap of less than 1.0.

## 🏆 Key Results
The final model, a Weighted Voting Regressor (Ridge + Random Forest + Gradient Boosting), achieved "Holy Grail" performance metrics:

| Metric | Score |
| :--- | :--- |
| **Test MSE** | **182.09** |
| **Test RMSE** | **13.49 lbs** |
| **R² Score** | **0.87** |
| **Generalization Gap** | **0.97** (Virtually Zero Overfitting) |

## 🧠 Methodology: Physics-Informed Features
The core innovation is the **"Physics Proxy"** feature. Instead of forcing the model to learn physiological rules from scratch, we engineered a baseline estimate using the standard mass formula:

$$Weight \approx BMI \times Height^2$$

Since the dataset provided BMI only as categorical ranges (Underweight, Normal, Obese), we mapped these categories to their **clinical arithmetic midpoints** based on CDC/WHO standards (e.g., Normal Range 18.5-24.9 $\rightarrow$ 21.7). This transformed a raw classification problem into a residual correction task, significantly reducing variance.

## 🛠️ Tech Stack
* **Python 3.9+**
* **Scikit-Learn:** Pipelines, ColumnTransformer, VotingRegressor
* **Pandas & NumPy:** Data manipulation and vectorization
* **Ensemble Methods:** Gradient Boosting, Random Forest, Ridge Regression

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/nhis-weight-prediction.git](https://github.com/yourusername/nhis-weight-prediction.git)
    cd nhis-weight-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    ```bash
    python train_model.py
    ```
    *This will train the ensemble on the full dataset and generate the submission `.pkl` file.*

## 📂 Project Structure
* `train_model.py`: Main script containing the PhysicsFeatureCreator class and training pipeline.
* `requirements.txt`: List of dependencies.
* `Javvad_LastName2_LastName3_model_info.pkl`: The final trained model artifact.

## 👥 Authors
* **Dawood Javvad**
* [Partner Name]
* [Partner Name]
