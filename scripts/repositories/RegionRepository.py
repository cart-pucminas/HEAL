from models.Region import Region
from utils.DatabaseUtil import DatabaseUtil
import pandas as pd

class RegionRepository(DatabaseUtil):
    _instance = None

    def deleteAllRegions(self):
        print("===============================")
        print("Deleting all Regions...")
        query = 'delete from regions;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Regions SUCCESS...")
        else:
            print("Deleting Regions ERROR")
        print("===============================")     

        return data_in_database

    def insertRegions(self, regions):
        print("===============================")
        print("Inserting " + str(len(regions)) + " Regions in database...")

        query = ''
        params = []
        for region in regions:
            query += 'INSERT INTO regions (region_id, name, code) VALUES (%s, %s, %s);'
            params += [region.region_id, region.name[0:100], region.code[0:10]]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Regions SUCCESS")
        else:
            print("Inserting Regions ERROR")
        print("===============================")     

        return data_in_database


    def findAllRegions(self) -> Region:
        query = 'SELECT * FROM regions;'
        data_in_database = self.findAll(query, None)
        regions = Region.build(data_in_database)
        regionsDF = pd.DataFrame(data_in_database, columns=['region_id', 'name', 'code'])
        return regions, regionsDF

