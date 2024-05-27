
class Config:
    __instance = None

    def __init__(self):

        # PREDICTION INDICATOR CONFIG
        # Choose the indicator to forecast (only one for each execution)
        # see the options in https://ghoapi.azureedge.net/api/Indicator
        self.codePredictionIndicator = "WSH_10"


        
        

        # PREDICTION INDICATOR DIMENSIONS CONFIG
        # TO NOT USE DIMENSION IN PREDICTION INDICATOR DIMENSIONS -> use self.predictionIndicatorDimensions = [] 
        # Choose dimensions to use during the forecast - see the options in https://ghoapi.azureedge.net/api/Dimension
        # Therefore, choose how the dimensions will be applied - see the options using:
        # https://ghoapi.azureedge.net/api/DIMENSION/dimensionName/DimensionValues, change dimensionName to chosen dimension
        # use "Code" to dimension and "Title" to filter the chosen dimension, for instance: self.predictionIndicatorDimensions = [['SEX', 'Both sexes']]
        self.predictionIndicatorDimensions = []


        # SERÁ ACRESCENTADO ->
        # 0 para não usar
        # TO NOT USE PREDICTION INDICATOR YEAR AND PREDICTION INDICATOR REGION, use:
        self.predictionIndicatorYear = 0
        self.predictionIndicatorRegion= 0
        # When not used, the system will process data from the world (world countries context)
        # self.predictionIndicatorYear = 2019
        # self.predictionIndicatorRegion= 'Asia'


        # INDICATOR DIMENSIONS CONFIG
        # see the options in https://ghoapi.azureedge.net/api/Dimension
        # use only the "Code", for instance: self.dimensionsPrediction = ['SEX', 'AGEGROUP', 'EDUCATIONLEVEL']
        self.dimensionsPrediction = []
        
        # PERCENT MISSING DATAS CONFIG
        self.maxPercentMissingDatasToPrediction = 50

        # INFORMATION GAIN CONFIG
        self.minInformationGain = 2

        # TOTAL INDICATORS USE TO PREDICTION CONFIG
        self.totalIndicatorsUseToPrediction = 20



        # INDICATORS ANALYZING CONFIG
        # if list null, all indicators will be consider
        # see the options in https://ghoapi.azureedge.net/api/Indicator
        # use only the "IndicatorCode", for instance: self.INDICATORS = ['HEMOGLOBINLEVEL_PREGNANT_MEAN', 'HEMOGLOBINLEVEL_NONPREGNANT_ME', 'AIR_11'] 
        self.indicatorsAnalyzingCode = []

        # TOTAL GROUPS FOR CATEGORIZATION CONFIG
        self.groupsCategorization = 5
        self.typeML = "classification"

        # OUTLIERS CONFIG
        # Q1 and Q3 Outliers parameters
        self.outlierQparameter = [0.1, 0.9]

        # FORECAST CONFIG
        # Database fraction used for traning algorithm to prediction. Database test will be 1 - fracUsedToTraining
        self.fracUsedToTraining = 0.6667
        

        # Pearson Correlation Cut
        # Pearson Correlation value used to remove indicators that are not indicate to use by a prediction system
        self.pearsonCorrelationCut = 0.7
        




    @staticmethod
    def instance():
        if not Config.__instance:
            Config.__instance = Config()
        return Config.__instance

config = Config.instance()

if __name__ == "__main__":
    s1 = Config.instance()
    s2 = Config.instance()
