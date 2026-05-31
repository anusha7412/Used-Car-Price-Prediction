##  Project Overview

This project predicts the selling price of used cars using Machine Learning techniques. The model is trained on the CarDekho dataset and uses various car attributes such as brand, year, fuel type, transmission type, kilometers driven, and ownership details to estimate the resale value of a vehicle.

The objective of this project is to help buyers and sellers make informed decisions by providing an accurate prediction of a used car's market price.

##  Objectives

- Analyze and preprocess used car data.
- Perform exploratory data analysis (EDA).
- Train a Machine Learning model for price prediction.
- Evaluate model performance using regression metrics.
- Predict the selling price of a car based on user inputs.

##  Dataset

Dataset: CarDekho Used Car Dataset

The dataset contains information about various used cars, including:

- Car Name
- Year
- Selling Price
- Fuel Type
- Seller Type
- Transmission
- Owner
- Kilometers Driven

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Data Preprocessing

The following preprocessing steps were performed:

- Removed missing and duplicate values.
- Filtered out extreme outliers.
- Encoded categorical features using Label Encoding.
- Prepared the dataset for model training.

##  Machine Learning Model

Algorithm Used:

### Gradient Boosting Regressor

Gradient Boosting Regressor was selected because it provides high prediction accuracy and handles complex relationships between features effectively.

##  Model Evaluation

The model was evaluated using:

- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The trained model achieved strong prediction performance on the testing dataset.

##  Project Structure

Used-Car-Price-Prediction/

├── cardekho_dataset.csv

├── train_model.py

├── predict.py

├── model.pkl

├── label_encoders.pkl

├── requirements.txt

├── README.md

└── screenshots/

##  How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Used-Car-Price-Prediction.git
```

### Step 2: Navigate to Project Folder

```bash
cd Used-Car-Price-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_model.py
```

### Step 5: Run Predictions

```bash
python predict.py
```

##  Features Used for Prediction

- Car Brand
- Manufacturing Year
- Fuel Type
- Seller Type
- Transmission Type
- Owner Category
- Kilometers Driven

##  Screenshots

Add screenshots of:

- Dataset Preview
- Data Visualization
- Model Training Output
- Prediction Results

Store images inside the screenshots folder.

##  Future Enhancements

- Build a Streamlit web application.
- Deploy the model on cloud platforms.
- Add real-time car price estimation.
- Improve prediction accuracy using advanced ensemble methods.

##  Author

Anusha Balodiya

Artificial Intelligence & Machine Learning Enthusiast

##  Conclusion

This project demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and prediction. The Gradient Boosting Regressor model successfully predicts used car prices and can be further enhanced for real-world deployment.
