
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow




class Region:
    __instance = None

    def __init__(self, region_id: str = None, name: str = None, code: str = None):
        self._region_id = region_id
        self._name = name
        self._code = code

    @property
    def region_id(self) -> str:
        return self._region_id

    @region_id.setter
    def region_id(self, region_id):
        if region_id is not None and not isinstance(region_id, str):
            raise TypeError('invalid parameter format, type str')
        self._region_id = region_id

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

    



    @staticmethod
    def instance():
        if not Region.__instance:
            Region.__instance = Region()
        return Region.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        Region(data1['region_id'], 
                                data1['name'],
                                data1['code'],
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        Region(data1[0], 
                                data1[1],
                                data1[2],
                                )
                    )
                return returndata

            else:
                return Region(data1['region_id'], 
                                data1['name'],
                                data1['code'],
                                )

        else:
            return None
