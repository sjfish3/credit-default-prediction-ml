# Credit Card Default Prediction Using Machine Learning

This project applies machine learning models to predict whether a credit card client will default on their payment in the following month. The analysis uses the UCI Credit Card Default dataset, which contains financial and demographic information for 30,000 credit card clients.

The goal of the project is to explore the dataset, compare multiple predictive models (including Logistic Regression, Random Forest, and Gradient Boosting), and evaluate their ability to identify clients who are a default risk.

---

## Dataset

Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains:

- 30,000 credit card clients
- 25 features including demographic information, credit limits, billing amounts, payment history, and repayment status
- Binary target variable: **default payment next month**

Target variable encoding:
- `0` = No default
- `1` = Default

---

## Project Workflow

The project follows a typical machine learning workflow:

1. **Exploratory Data Analysis (EDA)**
   - Data structure inspection
   - Feature distributions
   - Class imbalance analysis

2. **Model Development**
   - Cross-validation
   - Feature scaling
   - Logistic Regression
   - Random Forest
   - Gradient Boosting

3. **Model Comparison**
   - ROC-AUC scoring

4. **Hyperparameter Tuning**
   - GridSearchCV used to optimize:
     - number of estimators
     - learning rate
     - maximum tree depth
    
5. **Final Model Evaluation**
   - ROC-AUC scoring
   - Precision / Recall analysis
   - Confusion matrix
   - ROC curve and Precision-Recall curve

  Below is the ROC curve for the final Gradient Boosting model evaluated on the test dataset.
  <img width="444" height="453" alt="image" src="https://github.com/user-attachments/assets/c5ab3827-a5ed-487d-99a1-744a402f4cbc" />

  
6. **Feature Importance Analysis**
   - Identification of key predictors that are most influential in the model

---

## Model Results

The **Gradient Boosting model** achieved the best performance:

- Cross-validated ROC-AUC: **~0.78**
- Test Accuracy: **81.8%**
- Precision: **66.4%**
- Recall: **35.9%**

The results highlight the impact of **class imbalance**, where accuracy can be misleading. ROC-AUC is strong at ~0.78 and measures the model's ability to rank clients who default as a higher risk than those who do not, regardless of classification thresholds. However, recall is fairly weak at 35.9%, indicating that the model incorrectly identified many defaulters. Recall is particularly important in credit risk modeling because failing to detect a borrower who will default can lead to significant financial losses. 

---

## Key Findings

- **Recent repayment status (PAY_0)** is by far the most influential predictor of default risk.
- Historical payment behavior strongly predicts future default.
- Demographic variables play a much smaller role compared to financial and repayment features.
- Adjusting classification thresholds could improve recall for real-world credit-default risk applications.

---

## Project Structure
│
├── credit_default_model.ipynb
├── README.md
├── data
  └── default_of_credit_card_clients.xls

---

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Future Improvements

Potential extensions of this project include:

- Additional feature engineering
- Expanded hyperparameter tuning
- Probability calibration
- Testing additional models
- Business-determined threshold optimization

---

## Author

Samantha Fish  
University of North Carolina at Chapel Hill  
Statistics & Mathematics
