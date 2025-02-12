# Portuguese Bank Deposit Marketing Data Analysis with Machine Learning

Conducting a machine learning analysis on Portuguese bank data to predict the likelihood of a customer subscribing to a term deposit.

## Summary

This project explores a dataset from a Portuguese bank, applying data preprocessing, feature engineering, and machine learning techniques. The goal is to build a predictive model to determine whether a customer will subscribe to a term deposit. Model evaluation is performed using accuracy, F1 score, and confusion matrix.

## Data

The dataset originates from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5K306). It contains various attributes about customers and past marketing campaigns, which are used to train predictive models.

## Approach

1. **Exploratory Data Analysis**

   - Load and inspect dataset structure (`.info()`, `.describe()`)
   - Identify categorical and numerical features
   - Check for missing values

2. **Data Visualization**

   - Analyze distributions of numerical features (age, balance, duration, etc.)
   - Compare target variable (`y`) across categorical variables
   - Detect outliers using boxplots

3. **Feature Engineering**
   - Drop irrelevant or highly imbalanced features (`default`, `pdays`)
   - Encode categorical variables using dummy encoding
   - Convert boolean columns (`housing`, `loan`) into binary format
4. **Handling Imbalanced Data**

   - Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution
   - Split data into training and testing sets

5. **Model Training & Evaluation**
   - Train models: `RandomForestClassifier` and `XGBClassifier`
   - Use cross-validation to compare model performance
   - Perform hyperparameter tuning with `GridSearchCV`
   - Select the best-performing model (XGBoost)
   - Evaluate model accuracy, F1-score, and confusion matrix
   - Analyze feature importance

## Built With

- Scikit-learn (SKLearn)
- Imbalanced-learn (SMOTE)
- XGBoost
- Pandas, Numpy, Matplotlib, Seaborn

## Authors

Ichsan Hibatullah

## Acknowledgments

Special thanks to the UCI Machine Learning Repository and the Kaggle community for providing data and insights.
