import pandas as pd
from pandas import json_normalize
from models.Dimension import Dimension
from repositories import DimensionRepository
from utils import Request, Utils

from config.ConfigCreateDatabase import config

def dimensionController():
    statusInsert = False
    dataTypes = []
    resultRequest = []
    statusRequest = False
    dimensionsDatasDF = pd.DataFrame(columns=['dimension_id', 'name', 'code'])
    dimensionsDatasFinal = []

    print('Searching for Dimensions in WHO API...')    
    dimensions = []
    if not config.DIMENSIONS:
        resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/Dimension")
        if resultRequest:
            resultRequest = resultRequest.json() 
            resultRequest = json_normalize(resultRequest['value'])
            resultRequest = resultRequest[resultRequest['Code'] != 'REGION']
            resultRequest = resultRequest[resultRequest['Code'] != 'YEAR']
            resultRequest = resultRequest[resultRequest['Code'] != 'COUNTRY']
            dimensions = resultRequest['Code'].values
    else:
        dimensions = config.DIMENSIONS

    for dimension in dimensions:
        print('---> Searching for Dimensions ' + str(dimension) + ' in WHO API...')    
        resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/DIMENSION/" + str(dimension) + "/DimensionValues")
        if resultRequest:
            resultRequest = resultRequest.json() 
            resultRequest = json_normalize(resultRequest['value'])
            if not resultRequest.empty:         
                resultRequest = Utils.createUuid(resultRequest.loc[:, ['Title', 'Code']])
                resultRequest.rename(columns={'id': 'dimension_id', 'Title': 'name', 'Code':'code'}, inplace = True)
                resultRequest['original_dimension'] = dimension
                dimensionsDatasDF = pd.concat([dimensionsDatasDF, resultRequest])
                dimensionsDatas = Dimension.build(resultRequest.loc[:, ['dimension_id', 'name', 'code', 'original_dimension']])
                dimensionsDatasFinal.extend(dimensionsDatas) 
                statusInsert = DimensionRepository.DimensionRepository().insertDimensions(dimensionsDatas)
                statusRequest = statusInsert
            else:
                print(str(dimension) + " doesn't contain values...")
                print("===============================")
        else:
            print(str(dimension) + " error...")
            print(resultRequest)
            
    return statusRequest, dimensionsDatasFinal, dimensionsDatasDF