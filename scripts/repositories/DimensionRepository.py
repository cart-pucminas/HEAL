from utils.DatabaseUtil import DatabaseUtil
from models.Dimension import Dimension
import pandas as pd

class DimensionRepository(DatabaseUtil):
    _instance = None

    def deleteAllDimension(self):
        print("===============================")
        print("Deleting all Dimensions...")
        query = 'delete from dimensions;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Dimensions SUCCESS...")
        else:
            print("Deleting Dimensions ERROR")
        print("===============================")     
        
        return data_in_database

    def insertDimensions(self, dimensions):
        print("===============================")
        print("Inserting " + str(len(dimensions)) + " Dimensions in database...")

        query = ''
        params = []
        for dimension in dimensions:
            query += 'INSERT INTO dimensions (dimension_id, name, code, original_dimension) VALUES (%s, %s, %s, %s);'
            params += [dimension.dimension_id, dimension.name[0:100], dimension.code[0:100], dimension.original_dimension[0:100]]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting Dimensions SUCCESS")
        else:
            print("Inserting Dimensions ERROR")
        print("===============================")     

        return data_in_database

    def findAllDimensions(self) -> Dimension:
        query = 'SELECT * FROM dimensions;'
        data_in_database = self.findAll(query, None)
        dimensions = Dimension.build(data_in_database)
        dimensionsDF = pd.DataFrame(data_in_database, columns=['dimension_id', 'name', 'code', 'original_dimension'])
        return dimensions, dimensionsDF

