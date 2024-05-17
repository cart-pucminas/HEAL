from utils.DatabaseUtil import DatabaseUtil
from models.Year import Year
import pandas as pd

class YearRepository(DatabaseUtil):
    _instance = None

    def deleteAllYears(self):
        print("===============================")
        print("Deleting all Years...")
        query = 'delete from years;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Years SUCCESS...")
        else:
            print("Deleting Years ERROR")
        print("===============================")     
        
        return data_in_database

    def insertYears(self, years):
        print("===============================")
        print("Inserting " + str(len(years)) + " Years in database...")

        query = ''
        params = []
        for year in years:
            query += 'INSERT INTO years (year_id, year, code) VALUES (%s, %s, %s);'
            params += [year.year_id, year.year[0:30], year.code[0:30]]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Years SUCCESS")
        else:
            print("Inserting Years ERROR")
        print("===============================")     

        return data_in_database

    
    def findAllYears(self) -> Year:
        query = 'SELECT * FROM years;'
        data_in_database = self.findAll(query, None)
        years = Year.build(data_in_database)
        yearsDF = pd.DataFrame(data_in_database, columns=['year_id', 'year', 'code'])
        return years, yearsDF

