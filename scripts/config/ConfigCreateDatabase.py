
class Config:
    __instance = None

    def __init__(self):
        # BASIC CONFIGS
        self.QTD_REQUEST = 15
        self.TIME_REQUEST = 2

        # DATABASE CONFIGS
        self.DATASOURCE_URL = 'localhost'
        self.DATASOURCE_DB = 'databasewhoprediction'
        self.DATASOURCE_USERNAME = 'userwhoprediction'
        self.DATASOURCE_PASSWORD = 'prmTW(fWY75JDgq!qqg(VnvCBfgpLmdKnbSq1CF'

        # DIMENSIONS CONFIG
        # if list null, all dimensions will be consider
        # see the options in https://ghoapi.azureedge.net/api/Dimension
        # use only the "Code", for instance: self.DIMENSIONS = ['AGEGROUP', 'EDUCATIONLEVEL', 'SEX']
        self.DIMENSIONS = ['AGEGROUP', 'SEX']
        
        # INDICATORS CONFIG
        # if list null, all indicators will be consider
        # see the options in https://ghoapi.azureedge.net/api/Indicator
        # use only the "IndicatorCode", for instance: self.INDICATORS = ['AIR_10', 'AIR_11', 'AIR_12']   
        self.INDICATORS = []        
        






    @staticmethod
    def instance():
        if not Config.__instance:
            Config.__instance = Config()
        return Config.__instance

config = Config.instance()

if __name__ == "__main__":
    s1 = Config.instance()
    s2 = Config.instance()