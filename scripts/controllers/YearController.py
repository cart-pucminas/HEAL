import pandas as pd
from pandas import json_normalize
from models.Year import Year
from repositories import YearRepository
from utils import Request, Utils

def yearController():
    statusInsert = False
    print('Searching for years (periods of time) in WHO API...')    
    resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/DIMENSION/YEAR/DimensionValues")

    if resultRequest:
        resultRequest = resultRequest.json() 
        resultRequest = json_normalize(resultRequest['value'])        
        resultRequest = Utils.createUuid(resultRequest.loc[:, ['Title', 'Code']])
        resultRequest.rename(columns={'id': 'year_id', 'Title': 'year', 'Code':'code'}, inplace = True)
        years = Year.build(resultRequest.loc[:, ['year_id', 'year', 'code']])
        statusInsert = YearRepository.YearRepository().insertYears(years)
    return statusInsert, years, resultRequest