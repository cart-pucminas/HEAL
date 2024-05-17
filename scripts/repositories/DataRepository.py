from utils.DatabaseUtil import DatabaseUtil
from models.Data import Data

import pandas as pd

class DataRepository(DatabaseUtil):
    _instance = None

    def deleteAllDatas(self):
        print("===============================")
        print("Deleting all Datas...")
        query = 'delete from datas;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Datas SUCCESS...")
        else:
            print("Deleting Datas ERROR")
        print("===============================")     
        
        return data_in_database

    def insertDatas(self, datas):
        print("===============================")
        print("Inserting " + str(len(datas)) + " Datas in database...")

        query = ''
        params = []
        for data in datas:
            query += 'INSERT INTO datas (data_id, value, country_id, indicator_id, year_id) VALUES (%s, %s, %s, %s, %s);'
            params += [data.data_id, data.value, data.country_id, data.indicator_id, data.year_id]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Datas SUCCESS")
        else:
            print("Inserting Datas ERROR")
        print("===============================")     

        return data_in_database

    # find information about data passing Indicator Name or Indicator Code
    def findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(self, indicatorName=None, indicatorCode=None):
        if indicatorName == None and indicatorCode == None:
            return False, False
        

        query = 'select d.data_id, y.year, c.name, r.name, d.value from datas d inner join years y on d.year_id = y.year_id inner join countries c on d.country_id = c.country_id inner join regions r on r.region_id = c.region_id inner join indicators i on d.indicator_id = i.indicator_id '
        if indicatorName != None and indicatorCode == None:
            query = query + 'where i.name = %s;'
            params = [indicatorName]
        elif indicatorName == None and indicatorCode != None:
            query = query + 'where i.code = %s;'
            params = [indicatorCode]
        else:
            query = query + 'where i.name = %s and i.code = %s;'
            params = [indicatorName, indicatorCode]

        data_in_database = self.findAll(query, params)
        # datas = Data.build(data_in_database)
        datasDf = pd.DataFrame(data_in_database, columns=['dataId', 'year', 'countryName', 'regionName', indicatorCode])
        return datasDf

    


     # find information about data passing Indicator Name or Indicator Code
    def findDataIdDimensionName(self, indicatorCode=None, originalDimension=None):
        if (indicatorCode == None) or (originalDimension == None):
            return False, False
        
        query = 'select d.data_id, di.name from datas_dimensions dd inner join dimensions di on dd.dimension_id = di.dimension_id inner join datas d on d.data_id = dd.data_id inner join indicators i on i.indicator_id = d.indicator_id '
        query = query + 'where i.code = %s and di.original_dimension = %s;'
        params = [indicatorCode, originalDimension]

        data_in_database = self.findAll(query, params)
        # datas = Data.build(data_in_database)
        datasDf = pd.DataFrame(data_in_database, columns=['dataId', originalDimension])
        return datasDf




