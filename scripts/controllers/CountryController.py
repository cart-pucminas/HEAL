import pandas as pd
from pandas import json_normalize
from models.Country import Country
from repositories import CountryRepository
from utils import Request, Utils

def countryController(regions):
    statusInsert = False
    countries = []
    regions.rename(columns={'code': 'ParentCode'}, inplace = True)
    regions = regions[['region_id', 'ParentCode']]
    
    print('Searching for contries in WHO API...')    
    resultRequest = Request.request_POST("https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues")

    if resultRequest:
        resultRequest = resultRequest.json() 
        resultRequest = json_normalize(resultRequest['value'])    
        resultRequest = pd.merge(resultRequest, regions, how="inner", on=["ParentCode"])
        resultRequest = Utils.createUuid(resultRequest.loc[:, ['Title', 'Code', 'region_id']])
        resultRequest.rename(columns={'id': 'country_id', 'Title': 'name', 'Code':'code'}, inplace = True)
        countries = Country.build(resultRequest.loc[:, ['country_id', 'name', 'code', 'region_id']])
        statusInsert = CountryRepository.CountryRepository().insertCountries(countries)
    return statusInsert, countries, resultRequest