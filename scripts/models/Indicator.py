
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow




class Indicator:
    __instance = None

    def __init__(self, indicator_id: str = None, name: str = None, code: str = None):
        self._indicator_id = indicator_id
        self._name = name
        self._code = code

    @property
    def indicator_id(self) -> str:
        return self._indicator_id

    @indicator_id.setter
    def indicator_id(self, indicator_id):
        if indicator_id is not None and not isinstance(indicator_id, str):
            raise TypeError('invalid parameter format, type str')
        self._indicator_id = indicator_id

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
        if not Indicator.__instance:
            Indicator.__instance = Indicator()
        return Indicator.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        Indicator(data1['indicator_id'], 
                                data1['name'],
                                data1['code'],
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        Indicator(data1[0], 
                                data1[1],
                                data1[2],
                                )
                    )
                return returndata

            else:
                return Indicator(data1['indicator_id'], 
                                data1['name'],
                                data1['code']
                                )

        else:
            return None
