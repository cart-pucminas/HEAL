import pandas as pd
from pandas import json_normalize
from models.DataDimension import DataDimension
from repositories import DataDimensionRepository
from utils import Request, Utils

from config.ConfigCreateDatabase import config


def insertDataDimensions(datasByDim, dimension_id):
    datasByDim['dimension_id'] = dimension_id
    resultRequest = Utils.createUuid(datasByDim.loc[:, ['data_id', 'dimension_id']])
    resultRequest.rename(columns={'id': 'data_dimension_id'}, inplace = True)
    dataDimensions = DataDimension.build(resultRequest.loc[:, ['data_dimension_id', 'data_id', 'dimension_id']])
    statusInsert = DataDimensionRepository.DataDimensionRepository().insertDataDimensions(dataDimensions)
    return statusInsert

def dataDimensionController(datas, dimensions):

    for dimension_id, dimensionCode, originalDimension in zip(dimensions['dimension_id'].values, dimensions['dimensionsCode'].values, dimensions['original_dimension'].values):
        i = 1; status = True
        while status:
            if (('Dim' + str(i)) in datas.columns) and (('Dim' + str(i) + 'Type') in datas.columns):
                datasDim = datas[datas['Dim' + str(i) + 'Type'].notna()]
                datasDim = datas[datas['Dim' + str(i)].notna()]
                datasDim = datasDim[datasDim['Dim' + str(i) + 'Type'] == originalDimension]
                datasDim = datasDim[datasDim['Dim' + str(i)] == dimensionCode]
                if not datasDim.empty:
                    statusInsert = insertDataDimensions(datasDim, dimension_id)
                i = i+1
            else:
                status = False


