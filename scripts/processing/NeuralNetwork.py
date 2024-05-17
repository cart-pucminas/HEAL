from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils import OversamplingUndersampling

from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from config.ConfigPrediction import config


def build_model(train_dataset):
      model = keras.Sequential([
            layers.Dense(63, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001), input_shape=[len(train_dataset.keys())]),
            keras.layers.Dropout(0.5),
            layers.Dense(63, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            layers.Dense(1)
            ])
      optimizer = tf.keras.optimizers.RMSprop(0.001)
      model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
      model.summary()
      return model

def build_model_categorical(train_dataset, max):
      model = keras.Sequential([
            layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001), input_shape=[len(train_dataset.keys())]),
            keras.layers.Dropout(0.5),
            layers.Dense(30, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dense(max)
            ])
      optimizer = tf.keras.optimizers.RMSprop(0.001)
      model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
      model.summary()
      return model


def ANN(dataset, output):
      outputFinal = dataset.pop(output)
      train_features, test_features, train_labels, test_labels = train_test_split(dataset, outputFinal, test_size=1-config.fracUsedToTraining, random_state=0)

      print(train_features)
      print(train_labels)
      train_features, train_labels = OversamplingUndersampling.oversampling(train_features, train_labels)
      print(train_features)
      print(train_labels)

      train_stats = train_features.describe()
      train_stats = train_stats.transpose()

      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])
      
      train_features = norm(train_features)
      test_features = norm(test_features)

      max = 0 
      for i in (train_labels.to_list()):
            if (int(i) > max):
                  max = int(i)
                  
      # model = KerasRegressor(model=build_model(train_features))
      model = KerasClassifier(model=build_model_categorical(train_features, max))
      
      space = dict()
      space['epochs'] = [10]
      # space['epochs'] = [10, 50, 100]
      # space['validation_split'] = [0.2, 0.4, 0.6]
      space['validation_split'] = [0.2]
            
      search = GridSearchCV(estimator=model, param_grid=space, scoring='explained_variance', cv=2, n_jobs=1)
      # search = GridSearchCV(estimator=model, param_grid=space, scoring='explained_variance', cv=5, n_jobs=1)
      result = search.fit(train_features, train_labels)
      best_model = result.best_estimator_
      
      resultado = pd.DataFrame.from_dict(result.cv_results_)[['mean_test_score', 'std_test_score', 'rank_test_score']]
      print("Todos os resultados encontrados nos testes")
      print(resultado)
      print("Melhor resultado (rank = 1)")
      print(resultado[resultado['rank_test_score'] == 1][['mean_test_score', 'std_test_score']])
                          
      # test_predictions = best_model.predict(test_features).flatten()
      test_predictions = best_model.predict(test_features).flatten()
      
      new_test_predictions = []
      for i in test_predictions:
            new_test_predictions.append(round(i))

      new_test_labels = []
      for j in test_labels:
            new_test_labels.append(round(j))

      result = pd.DataFrame()
      result['Actual Value'] = new_test_labels
      result['Prediction'] = new_test_predictions
      return result

def ANNByRegionYear(dataset, output, predictionIndicatorYear, predictionIndicatorRegion):
      dataset['year'] = dataset['year'].astype('string')
      dataset['regionName'] = dataset['regionName'].astype('string')
      test_dataset_byYear = dataset[dataset['year'] == str(predictionIndicatorYear)]
      test_dataset = test_dataset_byYear[test_dataset_byYear['regionName'] == str(predictionIndicatorRegion)]

      train_dataset = dataset.drop(test_dataset.index)
      train_dataset.pop('year')
      train_dataset.pop('regionName')
      test_dataset.pop('year')
      test_dataset.pop('regionName')

      train_stats = train_dataset.describe()
      train_stats.pop(output)
      train_stats = train_stats.transpose()

      train_labels = train_dataset.pop(output)
      test_labels = test_dataset.pop(output)
      
      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])

      train_features = norm(train_dataset)
      test_features = norm(test_dataset)

      model = KerasRegressor(model=build_model(train_features))
      
      space = dict()
      space['epochs'] = [10, 50, 100]
      space['validation_split'] = [0.2, 0.4]
            
      search = GridSearchCV(estimator=model, param_grid=space, scoring='explained_variance', cv=5, n_jobs=1)
      result = search.fit(train_features, train_labels)
      best_model = result.best_estimator_
      
      resultado = pd.DataFrame.from_dict(result.cv_results_)[['mean_test_score', 'std_test_score', 'rank_test_score']]
      print("Todos os resultados encontrados nos testes")
      print(resultado)
      print("Melhor resultado (rank = 1)")
      print(resultado[resultado['rank_test_score'] == 1][['mean_test_score', 'std_test_score']])
                          
      test_predictions = best_model.predict(test_features).flatten()
      
      new_test_predictions = []
      for i in test_predictions:
            new_test_predictions.append(round(i))

      new_test_labels = []
      for j in test_labels:
            new_test_labels.append(round(j))

      result = pd.DataFrame()
      result['Actual Value'] = new_test_labels
      result['Prediction'] = new_test_predictions

      return result

def main(raw_dataset, indicatorPredictionName, predictionIndicatorYear, predictionIndicatorRegion):
      print(raw_dataset)
      raw_dataset[indicatorPredictionName] = raw_dataset[indicatorPredictionName].astype(float)
      dataset = raw_dataset.copy()

      dataset.pop('countryName')

      if (str(predictionIndicatorYear) != '0' and str(predictionIndicatorRegion) != '0'):
            result = ANNByRegionYear(dataset, indicatorPredictionName, predictionIndicatorYear, predictionIndicatorRegion)
      else:
            random = randint(0,1000)
            dataset.pop('year')
            origin = dataset.pop('regionName')
            for region in raw_dataset['regionName'].unique():
                  dataset[region] = (origin == region)*1.0
            dataset.tail()
            result  = ANN(dataset, indicatorPredictionName)

      result = result.join(raw_dataset['countryName'])
      result = result.join(raw_dataset['regionName'])
      result = result.join(raw_dataset['year'])
      return result


