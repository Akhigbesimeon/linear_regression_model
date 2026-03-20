# Student GPA Predictor
# Mission
The goal of this project is to build and deploy a regression model that predicts a college student's GPA based on their daily habits, study patterns, and behavioral choices. The model will be integrated into a student wellness and academic coaching app I'm building, one that helps students figure out which habits are quietly dragging their grades down, and which ones are worth actually investing in.

# Data Source
College Students Habits & Performance Dataset from Kaggle, a large dataset of 1,000,000 student records covering behavioral, lifestyle, and academic habit metrics. Features include study hours, screen time, concentration score, procrastination score, backlogs, part-time work hours, and more.

# Model Implementation
This project implements ALL required regression models:

* Linear Regression (Baseline implementation)
* Gradient Descent (SGDRegressor with loss curve visualization)
* Decision Trees
* Random Forest

Random Forest achieved the strongest performance due to the non-linear relationships between student habits and GPA. Linear Regression serves as the interpretable baseline.
