import pandas as pd
from pandas import json_normalize
from utils import Request, Utils
from repositories import RegionRepository
from models.Region import Region


def regionController():
    print('Searching for regions in WHO API...')    
    resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/DIMENSION/REGION/DimensionValues")

    if resultRequest:
        statusInsert = False
        resultRequest = resultRequest.json() 
        resultRequest = json_normalize(resultRequest['value'])        
        resultRequest = Utils.createUuid(resultRequest.loc[:, ['Title', 'Code']])
        
        resultRequest.rename(columns={'id': 'region_id', 'Title': 'name', 'Code':'code'}, inplace = True)
        regions = Region.build(resultRequest.loc[:, ['region_id', 'name', 'code']])
        statusInsert = RegionRepository.RegionRepository().insertRegions(regions)
    return statusInsert, regions, resultRequest