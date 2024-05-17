import uuid
from config.ConfigPrediction import config
import pandas as pd

def createUuid(df):
    lista = []    
    for i in range(0, len(df.index), 1):
        lista.append(str(uuid.uuid4()))
    df['id'] = lista
    return df

def joinDimensionsDatas(dataAnalyzing, indicatorAnalyzingCode):
    datasFinal = []
    dataAnalyzing[indicatorAnalyzingCode] = dataAnalyzing[indicatorAnalyzingCode].astype(float)
    for region in dataAnalyzing['regionName'].unique():
        dataAnalyzing_byRegion = dataAnalyzing[dataAnalyzing['regionName'] == region]
        for country in dataAnalyzing['countryName'].unique():
            dataAnalyzing_byRegion_byCountry = dataAnalyzing_byRegion[dataAnalyzing_byRegion['countryName'] == country]
            for year in dataAnalyzing['year'].unique():
                dataAnalyzing_byRegion_byCountry_byYear = dataAnalyzing_byRegion_byCountry[dataAnalyzing_byRegion_byCountry['year'] == year]
                if not dataAnalyzing_byRegion_byCountry_byYear.empty:
                    datasFinal.append([year, country, region, dataAnalyzing_byRegion_byCountry_byYear[indicatorAnalyzingCode].sum()])
    dataAnalyzing = pd.DataFrame(datasFinal, columns=['year', 'countryName', 'regionName', indicatorAnalyzingCode])
    return dataAnalyzing

def createGroups(datasToSendToPrediction, category):
    max = datasToSendToPrediction['percent'].max()
    min = datasToSendToPrediction['percent'].min()
    datasToSendToPrediction['range'] = 0
    step = (max-min)/category
    start = min
    end = min + step
    resultDatabase = pd.DataFrame()
    i = 1
    while i <= category:
        if i == category:
            end = max + 1
            
        saida = datasToSendToPrediction[(datasToSendToPrediction['percent'] >= float(start)) & (datasToSendToPrediction['percent'] < float(end))]
        if not saida.empty:
            saida['range'] = i
            resultDatabase = resultDatabase.append(saida, sort=False)
        start = start + step
        end = end + step
        i = i + 1
    return resultDatabase
