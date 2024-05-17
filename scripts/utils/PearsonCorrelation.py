
import seaborn as sn
import matplotlib.pyplot as plot

from config.ConfigPrediction import config


def main(resultDatabase, code):
    correlationCut = config.pearsonCorrelationCut
    correlation = resultDatabase.corr()

    totalValuesCut = 0
    listValuesCut = []

    for dataLine, index in zip(correlation[code].values, correlation[code].index):
        if (index != code and (float(dataLine) > correlationCut or float(dataLine) < -correlationCut)):
            totalValuesCut = totalValuesCut + 1
            listValuesCut.append(index)
            resultDatabase.pop(index)

    print("Number of instances deleted using Pearson Correlation: " + str(totalValuesCut))
    print("Instances deleted: ")
    print(listValuesCut)

    # sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
    # plot.show()
    # plot.savefig('CorrelationEx1.png')

    return correlation, resultDatabase