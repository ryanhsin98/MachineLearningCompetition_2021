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
  -  <table>
  <tr>
    <td>項次</td>
    <td>品名</td>
    <td>描述</td>
  </tr>
  <tr>
    <td>1</td>
    <td>iPhone 5</td>
    <td>iPhone 5是由蘋果公司開發的觸控式螢幕智慧型手機，是第六代的iPhone和繼承前一代的iPhone 4S。這款手機的設計比較以前產品更薄、更輕，及擁有更高解析度及更廣闊的4英寸觸控式螢幕，支援16:9寬螢幕。這款手機包括了一個自定義設計的ARMv7處理器的蘋果A6的更新、iOS 6操作系統，並且支援高速LTE網路。</td>
  </tr>
</table>
  -  | --------- | Public Leaderboard | | Private Leaderboard |
     | --------- | -------------------| | ------------------  |
     |  F1-score |      0.83832       | |       0.81673       |
     |    Rank#  |         #8         | |          #1         |
