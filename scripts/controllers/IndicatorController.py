import pandas as pd
from pandas import json_normalize
from models.Indicator import Indicator
from repositories import IndicatorRepository
from utils import Request, Utils

from config.ConfigCreateDatabase import config

def indicatorController():
    statusInsert = False
    indicators = []
    resultRequest = []

    statusRequest = False
    indicatorsDatasDF = pd.DataFrame(columns=['indicator_id', 'name', 'code'])
    indicatorsDatasFinal = []

    print('Searching for Indicators in WHO API...')    

    if not config.INDICATORS:
        resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/Indicator")
        if resultRequest:
            resultRequest = resultRequest.json() 
            resultRequest = json_normalize(resultRequest['value'])
            resultRequest = Utils.createUuid(resultRequest.loc[:, ['IndicatorName', 'IndicatorCode']])
            resultRequest.rename(columns={'id': 'indicator_id', 'IndicatorName': 'name', 'IndicatorCode':'code'}, inplace = True)
            indicatorsDatasDF = pd.concat([indicatorsDatasDF, resultRequest])
            indicatiorsDatas = Indicator.build(resultRequest.loc[:, ['indicator_id', 'name', 'code']])
            indicatorsDatasFinal.extend(indicatiorsDatas) 
            statusInsert = IndicatorRepository.IndicatorRepository().insertIndicators(indicatiorsDatas)
            statusRequest = statusInsert
    else:
        indicators = config.INDICATORS

    for indicator in indicators:
        print("---> Searching for Indicator " + str(indicator) + " in WHO API...")
        resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/Indicator/" + str(indicator))
        if resultRequest:
            resultRequest = resultRequest.json() 
            resultRequest = json_normalize(resultRequest)
            resultRequest = Utils.createUuid(resultRequest.loc[:, ['IndicatorName', 'IndicatorCode']])
            resultRequest.rename(columns={'id': 'indicator_id', 'IndicatorName': 'name', 'IndicatorCode':'code'}, inplace = True)
            indicatorsDatasDF = pd.concat([indicatorsDatasDF, resultRequest])
            indicatiorsDatas = Indicator.build(resultRequest.loc[:, ['indicator_id', 'name', 'code']])
            indicatorsDatasFinal.extend(indicatiorsDatas) 
            statusInsert = IndicatorRepository.IndicatorRepository().insertIndicators(indicatiorsDatas)
            statusRequest = statusInsert
        else:
            print(str(indicators) + " error...")
            print("===============================")
    
            
    return statusRequest, indicatorsDatasFinal, indicatorsDatasDF


def insertPopulation():
    details = {'name': ['Population'], 
                'code': ['POP']
            }
    resultRequest = pd.DataFrame(data=details)
    resultRequest = Utils.createUuid(resultRequest)
    resultRequest.rename(columns={'id': 'indicator_id'}, inplace = True)
    indicatiorsDatas = Indicator.build(resultRequest.loc[:, ['indicator_id', 'name', 'code']])
    statusInsert = IndicatorRepository.IndicatorRepository().insertIndicators(indicatiorsDatas)
    return statusInsert, indicatiorsDatas, resultRequest



