# CodeAlpha - Credit Scoring Model

## Task
Credit Scoring Model - Internship Task 1.

## Objective
Predict whether an individual is creditworthy using past financial information.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Logistic Regression

## Dataset
`dataset.csv` contains 1,200 synthetic records with:
- Age
- Annual Income
- Loan Amount
- Debt
- Credit History
- Payment History
- Employment Years
- Creditworthy (target)

The dataset is synthetic and intended for educational/internship demonstration.

## Project Workflow
1. Load the dataset.
2. Perform EDA and check missing values.
3. Visualize important features.
4. Preprocess numerical and categorical features.
5. Split data into training and testing sets.
6. Train a Logistic Regression classifier.
7. Evaluate using Accuracy, Precision, Recall, F1-Score and ROC-AUC.
8. Generate confusion matrix and ROC curve.
9. Test the model with a sample applicant.

## How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run EDA:

```bash
python EDA.py
```

Run the machine learning model:

```bash
python credit_scoring.py
```

The generated charts will be saved in the `results` folder.

## Expected Output
The program prints:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Classification report
- Sample applicant prediction

## Internship Submission
Suggested GitHub repository name:

`CodeAlpha_Credit_Scoring_Model`

Upload the complete project folder, then use the GitHub repository link in the LinkedIn project post and internship submission form.
