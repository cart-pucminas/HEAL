
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow




class DataDimension:
    __instance = None

    def __init__(self, data_dimension_id: str = None, data_id: str = None, dimension_id: str = None):
        self._data_dimension_id = data_dimension_id
        self._data_id = data_id
        self._dimension_id = dimension_id

    @property
    def data_dimension_id(self) -> str:
        return self._data_dimension_id

    @data_dimension_id.setter
    def data_dimension_id(self, data_dimension_id):
        if data_dimension_id is not None and not isinstance(data_dimension_id, str):
            raise TypeError('invalid parameter format, type str')
        self._data_dimension_id = data_dimension_id

    @property
    def data_id(self) -> str:
        return self._data_id

    @data_id.setter
    def data_id(self, data_id):
        if data_id is not None and not isinstance(data_id, str):
            raise TypeError('invalid parameter format, type str')
        self._data_id = data_id

    @property
    def dimension_id(self) -> str:
        return self._dimension_id

    @dimension_id.setter
    def dimension_id(self, dimension_id):
        if dimension_id is not None and not isinstance(dimension_id, str):
            raise TypeError('invalid parameter format, type str')
        self._dimension_id = dimension_id



    @staticmethod
    def instance():
        if not DataDimension.__instance:
            DataDimension.__instance = DataDimension()
        return DataDimension.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        DataDimension(data1['data_dimension_id'],
                                data1['data_id'],
                                data1['dimension_id']
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        DataDimension(data1[0], 
                                data1[1],
                                data1[2],
                                )
                    )
                return returndata

            else:
                return DataDimension(data1['data_dimension_id'],
                                data1['data_id'],
                                data1['dimension_id']
                                )

        else:
            return None
