import time
import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import classification_report
from tpot import TPOTClassifier

def testTrain(dataset, y):
    dataset = dataset.sort_values(by='year', ascending=True) 
    timeDimension = 'year'
    fracUsedToTraining = 0.7
    split_index = int(len(dataset) * fracUsedToTraining)
    train_val_df = dataset.iloc[:split_index]  
    test_df = dataset.iloc[split_index:]  
    X_test = test_df.drop(columns=[y, timeDimension])
    y_test = test_df[y]
    y_train = train_val_df[y]
    X_train = train_val_df.drop(columns=[y]) 
    if 'countryName' in X_train.columns:
        X_train.pop('countryName')
    if timeDimension in X_train.columns:
        X_train.pop(timeDimension)
    if 'regionName' in X_train.columns:
        X_train.pop('regionName')
    if 'countryName' in X_test.columns:
        X_test.pop('countryName')
    if timeDimension in X_test.columns:
        X_test.pop(timeDimension)
    if 'regionName' in X_test.columns:
        X_test.pop('regionName')

    return X_train, X_test, y_train, y_test


def tpot(path, y, listResult, maxTime, maxModels):
    dataset = pd.read_csv(path, sep=';', encoding='utf-8')
    X_train, X_test, y_train, y_test = testTrain(dataset, y)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    tpot = TPOTClassifier(
        generations=5,                   
        population_size=20,             
        verbosity=2,                     
        random_state=42,                
        max_time_mins=maxTime/60,           
        max_eval_time_mins=30,            
    )

    tpot.fit(X_train, y_train)

    predictions = tpot.predict(X_test)

    best_pipeline = tpot.fitted_pipeline_
    
    print("\nPipeline final escolhido pelo TPOT:")
    print(best_pipeline)

    final_model = best_pipeline.steps[-1][1]  # O último passo do pipeline é o modelo
    print("Modelo final:", final_model)

    print("Hiperparâmetros do modelo final:", final_model.get_params())

    print("Acurácia:", tpot.score(X_test, y_test))

    report_dict = classification_report(y_test, predictions, output_dict=True)
    print("\nRelatório de métricas:")
    print(report_dict)

    precisionTotal = 0
    recallTotal = 0
    f1Total = 0
    qtd = 0
    for class_label, metrics in report_dict.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:  # Ignorar médias e acurácia geral
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            listResult.append(['TPOT', f'class {class_label}', ' ', precision, recall, f1])
            precisionTotal = precisionTotal + precision
            recallTotal = recallTotal + recall
            f1Total = f1Total + f1
            qtd = qtd + 1

    precision = precisionTotal/qtd
    recall = recallTotal/qtd
    f1 = f1Total/qtd
    listResult.append(['TPOT', 'total', ' ', precision, recall, f1])

    return listResult, final_model, final_model.get_params()


def H2O(path, y, listResult, maxTime, maxModels):
    dataset = pd.read_csv(path, sep=';', encoding='utf-8')
    X_train, X_test, y_train, y_test = testTrain(dataset, y)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(train_df)
    h2o.init()
    train = h2o.H2OFrame(train_df)
    test = h2o.H2OFrame(test_df)

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    features = train.columns
    features.remove(y)

    aml = H2OAutoML(max_runtime_secs=maxTime, max_models=maxModels, seed=42)
    aml.train(x=features, y=y, training_frame=train)

    best_model = aml.leader
    predictions = best_model.predict(test)

    print("Predições no conjunto de teste:")
    print(predictions.head())

    performance = best_model.model_performance(test)
    print("Métricas no conjunto de teste:")
    print(performance)

    conf_matrix = performance.confusion_matrix()
    
    precisionTotal = 0
    recallTotal = 0
    f1Total = 0
    qtd = 0

    conf_matrix_df = conf_matrix.as_data_frame()

    print("Estrutura da matriz de confusão:")
    print(conf_matrix_df.columns)

    conf_matrix_df = conf_matrix_df.iloc[:-1]
    
    conf_matrix_df = conf_matrix_df.drop(columns=['Error', 'Rate'])

    print("Matriz de Confusão:")
    print(conf_matrix)

    class_labels = conf_matrix_df.columns

    print(class_labels)
    for idx, class_label in enumerate(class_labels):
        tp = conf_matrix_df.iloc[idx, idx]  # Diagonal principal
        fp = conf_matrix_df.iloc[:, idx].sum() - tp  # Soma da coluna menos TP
        fn = conf_matrix_df.iloc[idx, :].sum() - tp  # Soma da linha menos TP

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
        
        print(f"\nClasse: {class_label}")
        print(f" - Precisão: {precision}")
        print(f" - Recall: {recall}")
        print(f" - F1-Score: {f1}")
        listResult.append(['H2O', 'class ' + str(class_label), ' ', precision, recall, f1])
        precisionTotal = precisionTotal + precision
        recallTotal = recallTotal + recall
        f1Total = f1Total + f1
        qtd = qtd + 1

    precision = precisionTotal/qtd
    recall = recallTotal/qtd
    f1 = f1Total/qtd
    listResult.append(['H2O', 'total', ' ', precision, recall, f1])

    print("Melhor modelo escolhido:")
    print(best_model.algo)  
    print("Hiperparâmetros do melhor modelo:")
    print(best_model.get_params())

    h2o.shutdown(prompt=False)

    return listResult, best_model.algo, best_model.get_params()

outputColumns = ["HIV_0000000001", "NUTRITION_ANAEMIA_CHILDREN_PRE", "WSH_SANITATION_BASIC", "MH_12-Male"]
names = ['Ex1', 'Ex2', 'Ex3', 'Ex4']
maxTime = 6000
maxModels = 100


totalTimeList = []
listModels = []
for outputColumn, name in zip (outputColumns, names):

    path = str(name) + '.csv'
    listResult = []
    
    startTime = time.time()
    listResult, model, hiperparametro = H2O(path, outputColumn, listResult, maxTime, maxModels)
    listModels.append([name, 'H2O', model, hiperparametro])
    totalTime = time.time()-startTime
    print("Total time of H2O AutoML: " + str(totalTime) + " sec. (" + str((totalTime/60)/60) + " hour)")
    totalTimeList.append([name, 'H2O', totalTime])

    startTime = time.time()
    listResult, model, hiperparametro = tpot(path, outputColumn, listResult, maxTime, maxModels)
    listModels.append([name, 'tpot', model, hiperparametro])
    totalTime = time.time()-startTime
    print("Total time of TPOT AutoML: " + str(totalTime) + " sec. (" + str((totalTime/60)/60) + " hour)")
    totalTimeList.append([name, 'TPOT', totalTime])

    result = pd.DataFrame(listResult, columns=['AutoML', 'class', 'accuracy', 'precision', 'recall', 'f1-score'])
    print(result)
    result.to_csv('otherAutoMLResults_' + str(name) + '.csv' , sep=';',  encoding='utf-8')

listModels = pd.DataFrame(listModels, columns = ['name', 'AutoML', 'Modelo', 'Hiperparametro'])
listModels.to_csv('models.csv' , sep=';',  encoding='utf-8')

totalTimeList = pd.DataFrame(totalTimeList, columns=['name', 'AutoML', 'total time'])
totalTimeList.to_csv('totalTime.csv' , sep=';',  encoding='utf-8')
