from utils.DatabaseUtil import DatabaseUtil
from models.Indicator import Indicator
import pandas as pd

class IndicatorRepository(DatabaseUtil):
    _instance = None

    def deleteAllIndicators(self):
        print("===============================")
        print("Deleting all Indicators...")
        query = 'delete from indicators;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Indicators SUCCESS...")
        else:
            print("Deleting Indicators ERROR")
        print("===============================")     
        
        return data_in_database

    def insertIndicators(self, indicators):
        print("===============================")
        print("Inserting " + str(len(indicators)) + " Indicators in database...")

        query = ''
        params = []
        for indicator in indicators:
            query += 'INSERT INTO indicators (indicator_id, name, code) VALUES (%s, %s, %s);'
            params += [indicator.indicator_id, indicator.name[0:300], indicator.code[0:30]]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Indicators SUCCESS")
        else:
            print("Inserting Indicators ERROR")
        print("===============================")     

        return data_in_database


    def findAllIndicators(self) -> Indicator:
        query = 'SELECT * FROM indicators;'
        data_in_database = self.findAll(query, None)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF

    def findAllIndicatorsNotName(self, name) -> Indicator:
        query = 'SELECT * FROM indicators where name <> %s;'
        params = [name]
        data_in_database = self.findAll(query, params)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF

    def findAllIndicatorsNotCode(self, code) -> Indicator:
        query = 'SELECT * FROM indicators where code <> %s;'
        params = [code]
        data_in_database = self.findAll(query, params)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF

    
    def findAllIndicatorsNotCodeAndNotPOP(self, code) -> Indicator:
        query = 'SELECT * FROM indicators where code <> %s and code <> %s;'
        params = [code, 'POP']
        data_in_database = self.findAll(query, params)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF



    def findByName(self, name) -> Indicator:
        query = 'SELECT * FROM indicators where name = %s;'
        params = [name]
        data_in_database = self.findAll(query, params)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF

    def findByCode(self, code) -> Indicator:
        query = 'SELECT * FROM indicators where code = %s;'
        params = [code]
        data_in_database = self.findAll(query, params)
        indicators = Indicator.build(data_in_database)
        indicatorsDF = pd.DataFrame(data_in_database, columns=['indicator_id', 'name', 'code'])
        return indicators, indicatorsDF

