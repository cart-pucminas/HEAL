
import time

from controllers import RegionController, CountryController, YearController, DataController, DimensionController, IndicatorController
from repositories import RegionRepository, CountryRepository, YearRepository, DataRepository, DimensionRepository, IndicatorRepository, DataDimensionRepository


def main():
    startTime = time.time()
    DataDimensionRepository.DataDimensionRepository().deleteAllDataDimension()
    DataRepository.DataRepository().deleteAllDatas()
    YearRepository.YearRepository().deleteAllYears()
    CountryRepository.CountryRepository().deleteAllCountries()
    RegionRepository.RegionRepository().deleteAllRegions()
    DimensionRepository.DimensionRepository().deleteAllDimension()
    IndicatorRepository.IndicatorRepository().deleteAllIndicators()

    
    statusInsertYear, years, yearsDF = YearController.yearController()
    statusInsertRegion, regions, regionsDF = RegionController.regionController()

    if statusInsertRegion == True:
        statusInsertCountry, countries, countriesDF =CountryController.countryController(regionsDF)
    else:
        statusInsertCountry = False

    statusInsertDataTypes, dimensions, dimensionsDF = DimensionController.dimensionController()
    statusInsertIndicators, indicators, indicatorsDF  = IndicatorController.indicatorController()

    statusInsertIndicatorsPopulation, indicatorsPopulation, indicatorsDFPopulation  = IndicatorController.insertPopulation()
    DataController.insertPopulationDatas(yearsDF.copy(), countriesDF.copy(), dimensionsDF.copy(), indicatorsDFPopulation)

    if (statusInsertCountry == True) and (statusInsertYear == True) and (statusInsertDataTypes == True) and (statusInsertIndicators == True):
        statusInsertData = DataController.dataController(yearsDF, countriesDF, dimensionsDF, indicatorsDF)

    print("==============================================")
    print("==============================================")
    print("==============================================")
    print("Database created...")
    print("Total time: " + str(time.time()-startTime) + " sec.")

