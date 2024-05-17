from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from random import randint
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from config.ConfigPrediction import config

from utils import OversamplingUndersampling

from sklearn.metrics import r2_score


def RandomForestByRegionYear(dataset, output, predictionIndicatorYear, predictionIndicatorRegion):
      dataset['year'] = dataset['year'].astype('string')

      # test_dataset_byYear = dataset[dataset['year'] == str(predictionIndicatorYear)]
      test_features = dataset[dataset['year'] == str(predictionIndicatorYear)]
      # test_dataset = test_dataset_byYear[test_dataset_byYear[predictionIndicatorRegion] == 1]

      train_features = dataset.drop(test_features.index)
      train_features.pop('year')
      test_features.pop('year')

      train_labels = train_features.pop(output)
      test_labels = test_features.pop(output)

      train_stats = train_features.describe()
      # train_stats.pop(output)
      train_stats = train_stats.transpose()
      
      

      print(train_features)
      print(train_labels)
      # train_features, train_labels = OversamplingUndersampling.oversampling(train_features, train_labels)
      print(train_features)
      print(train_labels)
      
      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])
      train_features = norm(train_features)
      test_features = norm(test_features)
      # rf = RandomForestClassifier(random_state = 42)
      rf = RandomForestRegressor(random_state = 42)
      
      space = dict()
      space['n_estimators'] = [100, 110, 120]
      space['max_features'] = [1, 2, 4, 6, 8, 10, 12, 14, 16]
            
      # search = GridSearchCV(rf, space, scoring='accuracy', cv=5, refit=True)
      print(metrics.get_scorer_names())
      # 'neg_mean_absolute_error' -> - 0.79
      search = GridSearchCV(rf, space, cv=5, scoring='r2')
      # neg_root_mean_squared_error
      # neg_mean_absolute_percentage_error
      # r2

      result = search.fit(train_features, train_labels)
      print("Melhores parâmetros:")
      print(search.cv_results_['params'][search.best_index_])
      best_model = result.best_estimator_
      resultado = pd.DataFrame.from_dict(result.cv_results_)[['mean_test_score', 'std_test_score', 'rank_test_score']]
      
      print("Todos os resultados encontrados nos testes")
      print(resultado)
      print("Melhor resultado (rank = 1)")
      print(resultado[resultado['rank_test_score'] == 1][['mean_test_score', 'std_test_score']])
      predictions = best_model.predict(test_features)

      results = pd.DataFrame()
      results['Actual Value'] = test_labels
      results['Prediction'] = predictions
      return results


def RandomForest(dataset, output):
      outputFinal = dataset.pop(output)
      train_features, test_features, train_labels, test_labels = train_test_split(dataset, outputFinal, test_size=1-config.fracUsedToTraining, random_state=0)

      print(train_features)
      print(train_labels)
      # train_features, train_labels = OversamplingUndersampling.oversampling(train_features, train_labels)
      print(train_features)
      print(train_labels)

      train_stats = train_features.describe()
      train_stats = train_stats.transpose()

      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])
      
      train_features = norm(train_features)
      test_features = norm(test_features)
      # rf = RandomForestClassifier(random_state = 42)
      rf = RandomForestRegressor()
      
      space = dict()
      space['n_estimators'] = [100]
      space['max_features'] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
            
      # search = GridSearchCV(rf, space, scoring='accuracy', cv=5, refit=True)
      search = GridSearchCV(rf, space, cv=5, refit=True)
      result = search.fit(train_features, train_labels)
      print("Melhores parâmetros:")
      print(search.cv_results_['params'][search.best_index_])
      best_model = result.best_estimator_
      resultado = pd.DataFrame.from_dict(result.cv_results_)[['mean_test_score', 'std_test_score', 'rank_test_score']]
      
      print("Todos os resultados encontrados nos testes")
      print(resultado)
      print("Melhor resultado (rank = 1)")
      print(resultado[resultado['rank_test_score'] == 1][['mean_test_score', 'std_test_score']])
      predictions = best_model.predict(test_features)

      results = pd.DataFrame()
      results['Actual Value'] = test_labels
      results['Prediction'] = predictions
      return results

def main(raw_dataset, indicatorPredictionName, predictionIndicatorYear, predictionIndicatorRegion):
      raw_dataset[indicatorPredictionName] = raw_dataset[indicatorPredictionName].astype(float)
      dataset = raw_dataset.copy()
      dataset.pop('countryName')
      
      origin = dataset.pop('regionName')
      # for region in raw_dataset['regionName'].unique():
            # dataset[region] = (origin == region)*1.0
      # dataset.tail()
      
      if (str(predictionIndicatorYear) != '0' and str(predictionIndicatorRegion) != '0'):
            result = RandomForestByRegionYear(dataset, indicatorPredictionName, predictionIndicatorYear, predictionIndicatorRegion)
      
      else:
            dataset.pop('year')
            result = RandomForest(dataset, indicatorPredictionName)

      
      result = result.join(raw_dataset['countryName'])
      result = result.join(raw_dataset['regionName'])
      result = result.join(raw_dataset['year'])

      return result