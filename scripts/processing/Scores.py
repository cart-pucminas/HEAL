

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, mean_absolute_percentage_error, precision_score, recall_score
import math
import statistics
import scipy
import pandas as pd


def main(result):
    confusionMatrix  = pd.crosstab(result['Actual Value'], result['Prediction'], rownames=['Actual Value/Prediction'], colnames=['Prediction'])
    actual_values = result['Actual Value']
    prediction = result['Prediction']

    accuracy = (accuracy_score(actual_values,prediction))
    MSE = mean_squared_error(actual_values,prediction)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(actual_values,prediction)
    MAPE = mean_absolute_percentage_error(actual_values,prediction)

    precision = precision_score(actual_values, prediction, average="weighted")
    recall = recall_score(actual_values, prediction, average="weighted")
    f1 = 2*((precision*recall) / (precision+recall))

    # KGE calculate
    correlation = scipy.stats.pearsonr(actual_values, prediction)[0]
    desv = statistics.pstdev(prediction)/statistics.pstdev(actual_values)
    mean = statistics.mean(prediction)/statistics.mean(actual_values)
    KGE = ((1 - math.sqrt(math.pow(correlation-1,2) + math.pow(desv-1, 2) + math.pow(mean-1, 2))))
    # KGE calculate

    print("Accuracy: " + str(accuracy))
    print("MSE: " + str(MSE))
    print("RMSE: " + str(RMSE))
    print("MAE: " + str(MAE))
    print("MAPE: " + str(MAPE))
    print("KGE: " + str(KGE))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1))
    return accuracy, MSE, RMSE, MAE, MAPE, KGE, precision, recall, f1, confusionMatrix



def mainRegressor(result):
    actual_values = result['Actual Value']
    prediction = result['Prediction']

    MSE = mean_squared_error(actual_values,prediction)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(actual_values,prediction)
    MAPE = mean_absolute_percentage_error(actual_values,prediction)


    # KGE calculate
    correlation = scipy.stats.pearsonr(actual_values, prediction)[0]
    desv = statistics.pstdev(prediction)/statistics.pstdev(actual_values)
    mean = statistics.mean(prediction)/statistics.mean(actual_values)
    KGE = ((1 - math.sqrt(math.pow(correlation-1,2) + math.pow(desv-1, 2) + math.pow(mean-1, 2))))
    # KGE calculate

    print("MSE: " + str(MSE))
    print("RMSE: " + str(RMSE))
    print("MAE: " + str(MAE))
    print("MAPE: " + str(MAPE))
    print("KGE: " + str(KGE))


    return  0, MSE, RMSE, MAE, MAPE, KGE, 0, 0, 0, pd.DataFrame()