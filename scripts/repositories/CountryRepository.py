from utils.DatabaseUtil import DatabaseUtil
from models.Country import Country
import pandas as pd

class CountryRepository(DatabaseUtil):
    _instance = None

    def deleteAllCountries(self):
        print("===============================")
        print("Deleting all Countries...")
        query = 'delete from countries;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Countries SUCCESS...")
        else:
            print("Deleting Countries ERROR")
        print("===============================")     
        
        return data_in_database

    def insertCountries(self, countries):
        print("===============================")
        print("Inserting " + str(len(countries)) + " Countries in database...")

        query = ''
        params = []
        for country in countries:
            query += 'INSERT INTO countries (country_id, name, code, region_id) VALUES (%s, %s, %s, %s);'
            params += [country.country_id, country.name[0:100], country.code[0:20], country.region_id]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Countries SUCCESS")
        else:
            print("Inserting Countries ERROR")
        print("===============================")     

        return data_in_database

    def findAllCountries(self) -> Country:
        query = 'SELECT * FROM countries;'
        data_in_database = self.findAll(query, None)
        countries = Country.build(data_in_database)
        countriesDF = pd.DataFrame(data_in_database, columns=['country_id', 'name', 'code', 'region_id'])
        return countries, countriesDF

    def findCountryNameByIndicatorName(self, indicatorName) -> Country:
        query = 'select c.name from countries c inner join datas d on d.country_id = c.country_id inner join indicators i on d.indicator_id = i.indicator_id where i.name = %s;'
        params = [indicatorName]
        data_in_database = self.findAll(query, params)
        countriesDf = pd.DataFrame(data_in_database, columns=['name'])
        return countriesDf

    def findCountryNameByIndicatorCode(self, indicatorCode) -> Country:
        query = 'select c.name from countries c inner join datas d on d.country_id = c.country_id inner join indicators i on d.indicator_id = i.indicator_id where i.code = %s;'
        params = [indicatorCode]
        data_in_database = self.findAll(query, params)
        countriesDf = pd.DataFrame(data_in_database, columns=['name'])
        return countriesDf



