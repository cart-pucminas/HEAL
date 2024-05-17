
from repositories import CountryRepository

from config.ConfigPrediction import config
import pandas as pd

def missingDatasPercent(indicatorCode): 
    countries, countriesDf = CountryRepository.CountryRepository().findAllCountries()
    totalUniqueCountriesName = countriesDf['name'].unique()
    countries = CountryRepository.CountryRepository().findCountryNameByIndicatorCode(indicatorCode.replace("'", "''"))
    if not countries.empty:
        uniqueCountriesName = countries['name'].unique()
        result = 100 - ((len(uniqueCountriesName) / len(totalUniqueCountriesName)) * 100)
    else:
        result = 0
    return result

def missingDatasComplete(resultDatabase):
    indicatorsDatasMissingCompleted = []
    for column in resultDatabase.columns:
        try:
            percent = (resultDatabase[column].isnull().sum() / resultDatabase.shape[0])*100
            if percent > config.maxPercentMissingDatasToPrediction:
                resultDatabase = resultDatabase.drop([column], axis=1)
            else:
                dataFrequence = resultDatabase.columns.value_counts()[0]
                resultDatabase[column].fillna(dataFrequence, inplace=True)
                indicatorsDatasMissingCompleted.append([column, percent])
        except:
            print("Error trying complete missing datas")

    indicatorsDatasMissingCompleted = pd.DataFrame(indicatorsDatasMissingCompleted, columns=['indicatorAnalyzingName', 'missingDatasPercent'])

    return resultDatabase, indicatorsDatasMissingCompleted
   