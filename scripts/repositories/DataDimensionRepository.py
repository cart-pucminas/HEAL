from utils.DatabaseUtil import DatabaseUtil


class DataDimensionRepository(DatabaseUtil):
    _instance = None

    def deleteAllDataDimension(self):
        print("===============================")
        print("Deleting all datas_dimensions table datas...")
        query = 'delete from datas_dimensions;'
        data_in_database = self.delete_all(query)

        if data_in_database == True:
            print("Deleting Dimensions SUCCESS...")
        else:
            print("Deleting Dimensions ERROR")
        print("===============================")     
        
        return data_in_database

    def insertDataDimensions(self, dataDimensions):
        print("===============================")
        print("Inserting " + str(len(dataDimensions)) + " DataDimension in database...")

        query = ''
        params = []
        for datadimension in dataDimensions:
            query += 'INSERT INTO datas_dimensions (data_dimension_id, data_id, dimension_id) VALUES (%s, %s, %s);'
            params += [datadimension.data_dimension_id, datadimension.data_id, datadimension.dimension_id]

        data_in_database = self.save_or_update(query, params)   
        if data_in_database == True:
            print("Inserting DataDimensions SUCCESS")
        else:
            print("Inserting DataDimensions ERROR")
        print("===============================")     

        return data_in_database

