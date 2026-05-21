# 🏠 House Price Prediction Using Regression Models

**Predictive Insight Engine - Supervised Learning Project**  
**Red & White Skill Education | Duration: 6 Hours**

## 📌 Overview

This project builds an end-to-end supervised machine learning pipeline for house-price prediction in INR. The notebook explores a real-estate dataset, studies relationships between property features and price, compares regression models, demonstrates gradient descent from scratch, and converts model findings into business insights.

The project covers the full workflow:

```text
Data Preparation -> Simple Linear Regression -> Multiple Linear Regression
-> Polynomial Regression -> Gradient Descent -> Bias-Variance Diagnostics
-> Final Reporting
```

## 📁 Project Structure

```text
House-Price-Prediction-Project/
|
|-- My_House_Price_Prediction_Project.ipynb   # Main notebook: Parts A through I
|-- RealEstate_HousePrice_Dataset_4200.csv    # Dataset used by the notebook
`-- README.md                                 # Project documentation
```

## 📊 Dataset

The notebook uses:

```text
RealEstate_HousePrice_Dataset_4200.csv
```

The CSV file should be kept in the same folder as the notebook. The notebook can also read it from the local `Downloads` folder.

### 📋 Dataset Summary

| Property | Value |
| --- | --- |
| Total records | 4,200 houses |
| Total columns | 12 |
| Identifier columns | 1 |
| Feature columns | 10 |
| Target columns | 1 |
| Missing values | Dataset expected to be clean |
| Target | `house_price_inr` |

### 🧾 Feature Descriptions

| Column | Type | Role | Description |
| --- | --- | --- | --- |
| `house_id` | Integer | Identifier | Unique house ID, not used for modelling |
| `area_sqft` | Integer | Feature | Built-up property area in square feet |
| `bedrooms` | Integer | Feature | Number of bedrooms |
| `bathrooms` | Integer | Feature | Number of bathrooms |
| `location_score` | Float | Feature | Location desirability score |
| `age_years` | Integer | Feature | Age of the property in years |
| `distance_city_km` | Float | Feature | Distance from the city center in kilometres |
| `lot_size_sqft` | Integer | Feature | Total plot or lot size in square feet |
| `has_garage` | Integer | Feature | Garage availability: 1 = Yes, 0 = No |
| `has_pool` | Integer | Feature | Pool availability: 1 = Yes, 0 = No |
| `renovation_years_ago` | Integer | Feature | Years since the last renovation |
| `house_price_inr` | Integer | Target | House price in Indian Rupees |

### 📈 Main Feature Signals

The notebook uses scatter plots and a correlation heatmap to study which variables move with house price.

| Feature direction | Example interpretation |
| --- | --- |
| Positive | Larger area, more rooms, and a stronger location score can increase predicted price |
| Negative | Greater distance from the city and older property age can reduce predicted price |
| Binary amenities | Garage and pool indicators add context but may show weaker patterns in scatter plots |

## 🧩 Notebook Parts

## 🧠 Part A - Conceptual Understanding

The notebook answers six theory questions:

| Question | Topic | Key takeaway |
| --- | --- | --- |
| Q1 | Supervised Learning | Learns from labelled input-output examples |
| Q2 | Regression vs Classification | Regression predicts numeric values; classification predicts categories |
| Q3 | Simple Linear Regression | Models one feature and one continuous target |
| Q4 | Linear Regression Assumptions | Linearity, independence, residual behaviour, and low multicollinearity matter |
| Q5 | Bias-Variance Trade-off | Good generalization balances simplicity and flexibility |
| Q6 | Overfitting vs Underfitting | Underfitting misses patterns; overfitting memorizes training noise |

## 🔍 Part B - Dataset Understanding and Preparation

This part:

- Loads and previews the dataset
- Selects 10 modelling features and the house-price target
- Checks missing values, data types, and descriptive statistics
- Plots every selected feature against price
- Builds a correlation heatmap
- Splits data into training and testing sets

Train-test setup used in this notebook:

```text
Split ratio  : 80% training / 20% testing
Random seed  : random_state = 42
Target       : house_price_inr
```

## 📉 Part C - Simple Linear Regression

The first model uses only one feature:

```text
area_sqft -> house_price_inr
```

This baseline helps answer a simple question: how much of house-price variation can be explained by area alone?

The notebook:

- Trains a Simple Linear Regression model
- Prints slope and intercept
- Plots actual prices against the regression line
- Checks residual behaviour with diagnostic plots

## 📏 Part D - Model Evaluation Metrics

All regression models are compared using:

| Metric | What it measures | Better when |
| --- | --- | --- |
| MAE | Average absolute prediction error | Lower |
| MSE | Average squared prediction error | Lower |
| RMSE | Error magnitude in the same unit as price | Lower |
| R2 Score | Share of target variation explained by the model | Higher |

The baseline model is useful for comparison, but a one-feature price model is expected to leave many real-estate factors unexplained.

## 📚 Part E - Multiple Linear Regression

The main model uses all selected features together:

```text
area_sqft
bedrooms
bathrooms
location_score
age_years
distance_city_km
lot_size_sqft
has_garage
has_pool
renovation_years_ago
```

The notebook prints coefficients so each feature can be interpreted in business terms:

| Coefficient sign | Meaning |
| --- | --- |
| Positive | Higher feature value increases predicted price when other features are held constant |
| Negative | Higher feature value reduces predicted price when other features are held constant |

Multiple Linear Regression is the most practical model in this project because property value depends on several variables at the same time.

## 🧮 Part F - Polynomial Regression

Polynomial Regression is tested on `area_sqft` using degree 2 and degree 3 transformations.

| Model | Purpose |
| --- | --- |
| Degree 1 | Linear area-price baseline |
| Degree 2 | Tests a quadratic area-price relationship |
| Degree 3 | Tests a more flexible cubic area-price relationship |

The comparison shows whether increasing mathematical complexity on a single feature is enough, or whether the model needs more useful inputs.

## ⚙️ Part G - Gradient Descent Optimization

Gradient descent methods are implemented from scratch using NumPy.

### 🧰 Data Preparation for Gradient Descent

- Features are scaled with `StandardScaler`
- Target values are scaled to price in millions for numerical stability
- A bias column of ones is added to the feature matrix

### 🛠️ Implemented Methods

```python
# Batch Gradient Descent
def batch_gradient_descent(X_array, y_array, learning_rate=0.01, epochs=1000)

# Stochastic Gradient Descent
def stochastic_gradient_descent(X_array, y_array, learning_rate=0.01, epochs=50)

# Mini-batch Gradient Descent
def mini_batch_gradient_descent(
    X_array,
    y_array,
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
)
```

### 🔄 Gradient Descent Comparison

| Method | Data per update | Loss behaviour | Practical use |
| --- | --- | --- | --- |
| Batch GD | Full dataset | Smooth | Stable learning and debugging |
| Stochastic GD | One row | Noisier | Fast updates on large data |
| Mini-batch GD | Small batch | Balanced | Practical training workflow |

## 🩺 Part H - Bias-Variance and Diagnostics

The project compares model complexity and generalization behaviour:

| Model type | Main strength | Main risk |
| --- | --- | --- |
| Simple Linear Regression | Easy to interpret | High bias from using one feature |
| Polynomial Regression | Captures curved area patterns | Still limited if important features are missing |
| Multiple Linear Regression | Uses several property signals | Requires coefficient interpretation and feature checks |

The main diagnostic learning is that adding relevant features can improve a model more than increasing polynomial degree on a single feature.

## 📝 Part I - Final Analysis and Reporting

The final notebook section:

- Compares regression model results
- Identifies the most useful model for house-price estimation
- Summarizes gradient descent behaviour
- Explains feature impacts in business language
- Concludes why multiple-feature modelling is more useful than an area-only model

## 💼 Expected Business Insights

After running the notebook, the analysis helps explain:

- Why location quality can strongly influence price
- Why properties farther from the city may be valued lower
- Why age and renovation history matter
- Why house area is important but not enough on its own
- Why amenities add useful context to valuation

## 🚀 Setup and Usage

### 📦 Requirements

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### ▶️ Run the Notebook

```bash
jupyter notebook My_House_Price_Prediction_Project.ipynb
```

Then run the notebook cells from top to bottom.

Important: later notebook sections depend on variables created earlier during data loading, model training, evaluation, and gradient descent.

## 🛠️ Tech Stack

| Tool | Purpose |
| --- | --- |
| Python 3.x | Core programming language |
| NumPy | Matrix operations and gradient descent implementation |
| Pandas | Data loading and tabular analysis |
| Matplotlib | Visualizations and diagnostic plots |
| Seaborn | Correlation heatmap and plotting style |
| scikit-learn | Regression models, preprocessing, metrics, train-test split |
| Jupyter Notebook | Interactive machine learning workflow |

## ✅ Submission Checklist

- [x] 🧠 Part A - Theory answers
- [x] 🔍 Part B - Dataset loading, EDA, correlation analysis, and train-test split
- [x] 📉 Part C - Simple Linear Regression baseline and residual diagnostics
- [x] 📏 Part D - MAE, MSE, RMSE, and R2 evaluation
- [x] 📚 Part E - Multiple Linear Regression with coefficient analysis
- [x] 🧮 Part F - Polynomial Regression degree 2 and degree 3 comparison
- [x] ⚙️ Part G - Batch GD, SGD, and Mini-batch GD implemented from scratch
- [x] 🩺 Part H - Bias-variance discussion and model diagnostics
- [x] 📝 Part I - Final model interpretation and reporting
- [x] 🌐 GitHub-ready notebook and README

## 🎯 Conclusion

This project shows that house-price prediction is a supervised regression task where feature quality matters. A simple area-based baseline is easy to understand, but a multiple-feature model gives a stronger foundation for practical real-estate valuation.
