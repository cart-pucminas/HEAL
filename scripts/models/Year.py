
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow




class Year:
    __instance = None

    def __init__(self, year_id: str = None, year: str = None, code: str = None):
        self._year_id = year_id
        self._year = year
        self._code = code

    @property
    def year_id(self) -> str:
        return self._year_id

    @year_id.setter
    def year_id(self, year_id):
        if year_id is not None and not isinstance(year_id, str):
            raise TypeError('invalid parameter format, type str')
        self._year_id = year_id

    @property
    def year(self) -> str:
        return self._year

    @year.setter
    def year(self, year):
        if year is not None and not isinstance(year, str):
            raise TypeError('invalid parameter format, type str')
        self._year = year

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
        if not Year.__instance:
            Year.__instance = Year()
        return Year.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        Year(data1['year_id'], 
                                data1['year'],
                                data1['code'],
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        Year(data1[0], 
                                data1[1],
                                data1[2],
                                )
                    )
                return returndata

            else:
                return Year(data1['year_id'], 
                                data1['year'],
                                data1['code'],
                                )

        else:
            return None
