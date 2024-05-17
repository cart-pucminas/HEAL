import pandas as pd
from pandas import json_normalize
from models.Data import Data
from repositories import DataRepository
from utils import Request, Utils

from config.ConfigCreateDatabase import config

from controllers import DataDimensionController

def dataController(years, countries, dimensions, indicatorsDF):
    statusRequest = False
    datas = []
    countries.rename(columns={'code': 'SpatialDim'}, inplace = True)
    years.rename(columns={'code': 'TimeDim'}, inplace = True)
    dimensions.rename(columns={'code': 'dimensionsCode'}, inplace = True)

    total = len(indicatorsDF.index)
    actual = 1
    for indicator, indicatorId in zip(indicatorsDF['code'].values, indicatorsDF['indicator_id'].values):
        resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/" + str(indicator))
        if resultRequest:
            resultRequest = resultRequest.json() 
            resultRequest = json_normalize(resultRequest['value'])

            resultRequest['TimeDim'] = resultRequest['TimeDim'].astype('string')
            print(years)
            years['TimeDim'] = years['TimeDim'].astype('string')
            resultRequest = pd.merge(resultRequest, years, how="left", on=["TimeDim"])
            resultRequest = pd.merge(resultRequest, countries, how="left", on=["SpatialDim"])

            resultRequest = resultRequest[resultRequest['country_id'].notna()]
            resultRequest = resultRequest[resultRequest['year_id'].notna()]
            resultRequest = resultRequest[resultRequest['NumericValue'].notna()]
            resultRequest = Utils.createUuid(resultRequest.loc[:, ['country_id', 'year_id', 'NumericValue', 'Dim1Type', 'Dim1', 'Dim2Type', 'Dim2', 'Dim3Type', 'Dim3']])
            resultRequest['indicator_id'] = indicatorId
            resultRequest.rename(columns={'id': 'data_id', 'NumericValue': 'value'}, inplace = True)

            datas = Data.build(resultRequest.loc[:, ['data_id', 'value', 'country_id', 'indicator_id', 'year_id']])
            statusInsert = DataRepository.DataRepository().insertDatas(datas)
            statusRequest = statusInsert

            DataDimensionController.dataDimensionController(resultRequest, dimensions)

            print(str(round((actual/total)*100, 2)) + "% inserting Datas completed...")
            actual = actual+1

    return statusRequest


def insertPopulationDatas(years, countries, dimensions, indicatorsDF):

    datasPopulation = pd.read_csv('../files/WPP2022_TotalPopulationBySex.csv', sep=',', )
    datasPopulation = datasPopulation[datasPopulation['Variant'] == 'Medium']
    # df['PopMale'] = df['PopMale']*1000
    for nameColumn in ['PopMale', 'PopFemale', 'PopTotal', 'PopDensity']:
        datasPopulation[nameColumn] = datasPopulation[nameColumn].astype(float)
        datasPopulation[nameColumn] = datasPopulation[nameColumn] * 1000

    indicatorId = indicatorsDF['indicator_id'].values[0]
    print(datasPopulation[['ISO3_code', 'Location', 'Time', 'PopMale', 'PopFemale', 'PopTotal', 'PopDensity']])

    statusRequest = False
    datas = []
    countries.rename(columns={'name': 'Location'}, inplace = True)
    years.rename(columns={'code': 'Time'}, inplace = True)

    datasPopulation['Time'] = datasPopulation['Time'].astype('string')
    years['Time'] = years['Time'].astype('string')
    datasPopulation = pd.merge(datasPopulation, years, how="left", on=["Time"])
    datasPopulation = pd.merge(datasPopulation, countries, how="left", on=["Location"])

    datasPopulation = datasPopulation[datasPopulation['country_id'].notna()]
    datasPopulation = datasPopulation[datasPopulation['year_id'].notna()]
    datasPopulation = datasPopulation[datasPopulation['PopTotal'].notna()]
    datasPopulation = Utils.createUuid(datasPopulation.loc[:, ['country_id', 'year_id', 'PopTotal']])
    datasPopulation['indicator_id'] = indicatorId
    datasPopulation.rename(columns={'id': 'data_id', 'PopTotal': 'value'}, inplace = True)
    print(datasPopulation[['data_id', 'value', 'country_id', 'indicator_id', 'year_id']])
    datas = Data.build(datasPopulation.loc[:, ['data_id', 'value', 'country_id', 'indicator_id', 'year_id']])
    statusInsert = DataRepository.DataRepository().insertDatas(datas)
    statusRequest = statusInsert



    return statusRequest