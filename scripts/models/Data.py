
from numpy import isin
from pandas import DataFrame
from psycopg2.extras import DictRow


class Data:
    __instance = None

    def __init__(self, data_id: str = None, value: str = None, country_id: str = None, 
                indicator_id: str = None, year_id: str = None):
        self._data_id = data_id
        self._value = value
        self._country_id = country_id
        self._indicator_id = indicator_id
        self._year_id = year_id

    @property
    def data_id(self) -> str:
        return self._data_id

    @data_id.setter
    def data_id(self, data_id):
        if data_id is not None and not isinstance(data_id, str):
            raise TypeError('invalid parameter format, type str')
        self._data_id = data_id


    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError('invalid parameter format, type str')
        self._value = value

    
    @property
    def country_id(self) -> str:
        return self._country_id

    @country_id.setter
    def country_id(self, country_id):
        if country_id is not None and not isinstance(country_id, str):
            raise TypeError('invalid parameter format, type str')
        self._country_id = country_id

    @property
    def indicator_id(self) -> str:
        return self._indicator_id

    @indicator_id.setter
    def indicator_id(self, indicator_id):
        if indicator_id is not None and not isinstance(indicator_id, str):
            raise TypeError('invalid parameter format, type str')
        self._indicator_id = indicator_id

    @property
    def year_id(self) -> str:
        return self._year_id 

    @year_id.setter
    def year_id(self, year_id):
        if year_id is not None and not isinstance(year_id, str):
            raise TypeError('invalid parameter format, type str')
        self._year_id  = year_id


    @staticmethod
    def instance():
        if not Data.__instance:
            Data.__instance = Data()
        return Data.__instance


    @staticmethod
    def build(data: DictRow):

        if(data is not None):
            if(isinstance(data, list)):
                returndata = []
                for data1 in data:
                    returndata.append(
                        Data(data1['data_id'],
                                data1['value'],
                                data1['country_id'],
                                data1['indicator_id'],
                                data1['year_id']
                                )
                    )
                return returndata
            elif(isinstance(data, DataFrame)):
                returndata = []
                for data1 in data.values:
                    returndata.append(
                        Data(data1[0], 
                                data1[1],
                                data1[2],
                                data1[3],
                                data1[4]
                                )
                    )
                return returndata

            else:
                return Data(data1['data_id'],
                                data1['value'],
                                data1['country_id'],
                                data1['indicator_id'],
                                data1['year_id']
                                )

        else:
            return None
