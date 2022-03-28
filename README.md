# MachineLearningCompetition_2021
Private Leaderboard 1st place in Tsing hua University's Machine Learning Competition.

## Summary
The process of predicting the survival of Titanic passengers.
- Data pre-processing and feature selection：
  - missing value filling：
    - according to the characteristics of different data columns, more detailed missing value processing. Take "Pclass" for example, it takes is two steps to fill the missing values to maximize the approximation of actual passenger information. 
  - new feature creation：
    -  By observing the distribution of features and the correlation, six new features were created, accounting for up to 75% of the model’s feature set, with a view to effectively improve the model prediction performance.
- Model constuction：
  -  Choose the logistic regression model as the most suitable for this prediction task.
  -  The reason is that it is suitable for binary classification problems, and on the other hand, there is no data distribution assumptions like Naive bayes classifier which could occur the errors due to assumptions.
  -  Test a variety of feature combinations and tune ehe whole model formula (such as adding L2 regular term to the loss function).
-  Result：
  -  ranked first in the class on the Private Leaderboard.
  -  | --------- | Public Leaderboard | | Private Leaderboard |
     | --------- | -------------------| | ------------------  |
     |  F1-score |      0.83832       | |       0.81673       |
     |    Rank#  |         #8         | |          #1         |
