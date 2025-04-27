# Student Performance Prediction

This project uses the "StudentsPerformance" dataset to predict student performance based on various factors like gender, race/ethnicity, parental education, and lunch type. The goal is to perform both regression and classification tasks:
- **Regression**: Predict the student's math score.
- **Classification**: Classify students as pass/fail based on math score (threshold 50).

## Steps:
1. Preprocessing and encoding categorical variables.
2. Feature scaling.
3. Train a Linear Regression model for predicting math scores.
4. Train a Random Forest classifier for pass/fail prediction.
5. Evaluate models using R², RMSE, and accuracy scores.

## Requirements:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## How to Run:
1. Clone the repository: `git clone https://github.com/yourusername/Student-Performance-Prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python student_performance.py`
