
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow




class Dimension:
    __instance = None

    def __init__(self, dimension_id: str = None, name: str = None, code: str = None, original_dimension: str = None):
        self._dimension_id = dimension_id
        self._name = name
        self._code = code
        self._original_dimension = original_dimension

    @property
    def dimension_id(self) -> str:
        return self._dimension_id

    @dimension_id.setter
    def dimension_id(self, dimension_id):
        if dimension_id is not None and not isinstance(dimension_id, str):
            raise TypeError('invalid parameter format, type str')
        self._dimension_id = dimension_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        if name is not None and not isinstance(name, str):
            raise TypeError('invalid parameter format, type str')
        self._name = name

    @property
    def code(self) -> str:
        return self._code

    @code.setter
    def code(self, code):
        if code is not None and not isinstance(code, str):
            raise TypeError('invalid parameter format, type str')
        self._code = code

    @property
    def original_dimension(self) -> str:
        return self._original_dimension

    @original_dimension.setter
    def original_dimension(self, original_dimension):
        if original_dimension is not None and not isinstance(original_dimension, str):
            raise TypeError('invalid parameter format, type str')
        self._original_dimension = original_dimension

    



    @staticmethod
    def instance():
        if not Dimension.__instance:
            Dimension.__instance = Dimension()
        return Dimension.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        Dimension(data1['dimension_id'], 
                                data1['name'],
                                data1['code'],
                                data1['original_dimension']
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        Dimension(data1[0], 
                                data1[1],
                                data1[2],
                                data1[3]
                                )
                    )
                return returndata

            else:
                return Dimension(data1['dimension_id'], 
                                data1['name'],
                                data1['code'],
                                data1['original_dimension']
                                )

        else:
            return None
