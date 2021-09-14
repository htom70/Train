import logging
import config

import encoding
import flask
from flask import request, jsonify, request
import json

import time
from multiprocessing import Process, Queue, Manager
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer, RobustScaler, MaxAbsScaler, \
    MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.utils.validation import column_or_1d
from imblearn.under_sampling import RandomUnderSampler
import pika
import pickle
import database
import encoding
import engineering

app = flask.Flask(__name__)
app.config["DEBUG"] = True


def f(l):
    message = l[0]
    print('hello ' + message)


def getAllRecordsFromDatabase(databaseName, tableName):
    start = time.time()
    connection = database.getConnection()
    cursor = connection.cursor()
    sql_use_Query = "USE " + databaseName
    cursor.execute(sql_use_Query)
    sql_select_Query = f"select * from {tableName} order by timestamp"
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    numpy_array = np.array(result)
    end = time.time()
    elapsedTime = end - start
    print(f'{databaseName} beolvasva, betöltési idő: {elapsedTime}, rekordszám: {numpy_array.shape}')
    return numpy_array[:, :]


def getTrainParameters(trainTaskId):
    connection = database.getConnection()
    cursor = connection.cursor()
    sqlUseQuery = "USE mki"
    cursor.execute(sqlUseQuery)
    sqlSelectQuery = "select * from train_task where id = %s"
    parameter = (trainTaskId,)
    cursor.execute(sqlSelectQuery, parameter)
    result = cursor.fetchone()
    return result


def getSamplers():
    samplers = {
        'RandomUnderSampler': RandomUnderSampler(sampling_strategy=0.5),
        'RandomOverSampler': RandomOverSampler(sampling_strategy=0.5),
        'SMOTEENN': SMOTEENN(sampling_strategy=0.5, n_jobs=-1),
    }
    return samplers


#
#
def getScalers():
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer-Normal': QuantileTransformer(output_distribution='normal'),
        'QuantileTransformer-Uniform': QuantileTransformer(output_distribution='uniform'),
        'Normalizer': Normalizer(),
    }
    return scalers


#
#
def getFeatureSelectors():
    featureSelectors = {
        'RFE': RFECV(estimator=XGBClassifier(tree_method='gpu_hist', gpu_id=0), n_jobs=-1)
    }
    return featureSelectors


#
#
def getModels():
    models = {
        'Logistic Regression': LogisticRegression(n_jobs=-1),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB(),
        # 'SupportVectorMachine GPU': SupportVectorMachine(use_gpu=True),
        # 'Random Forest GPU': RandomForestClassifier(use_gpu=True, gpu_ids=[0, 1], use_histograms=True),
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        # 'MLP': MLPClassifier(),
        'Light GBM': LGBMClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    }
    return models


#
#
# def sendTrainMessage(id):
#     connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
#     channel = connection.channel()
#     channel.queue_declare(queue='train', durable=True)
#     message = {"trainId": id ,"result": "SUCCESFULLY"}
#     channel.basic_publish(exchange='', routing_key='train', body=json.dumps(message))
#     connection.close()
#
#
def persistEstmator(dataBaseName, estimatorName, estimatorContainer):
    connection = database.getConnection()
    cursor = connection.cursor()
    sqlUseQuery = "USE " + dataBaseName
    cursor.execute(sqlUseQuery)
    file = open("SQL INSERT estimator.txt", "r")
    sqlInsertQuery = file.read()
    cursor.execute(sqlInsertQuery, estimatorContainer)
    connection.commit()
    logging.info(f"Estimator: {estimatorName} persisted in database")

def fitAndStoreEstimator(parameters):
    logging.info("Fit process begin")
    estimatorStorePath = parameters.get('estimatorStorePath')
    pipelineId = parameters.get('trainTaskId')
    databaseName = parameters.get("databaseName")
    tableName = parameters.get("tableName")
    featureSelector = parameters.get("featureSelector")
    sampler = parameters.get("sampler")
    scaler = parameters.get("scaler")
    model = parameters.get("model")
    currencyEncoder = parameters.get("currencyEncoder")
    countryEncoder = parameters.get("countryEncoder")
    dataBaseHandler = database.DataBaseHandler(logging)
    dataset = dataBaseHandler.getAllRecordsFromDatabase(databaseName, tableName)
    features = dataset[:, 1:-1]
    labels = dataset[:, -1:]
    labels = labels.astype(int)
    labels = column_or_1d(labels)
    sampledFeatures, sampledLabels = sampler.fit_resample(features, labels)
    train_features, test_features, train_labels, test_labels = train_test_split(sampledFeatures, sampledLabels,
                                                                                test_size=0.2, random_state=0)
    pipeline = Pipeline(
        [('scaler', scaler), ('featureSelector', featureSelector), ('model', model)]
    )
    pipeline.fit(train_features, train_labels)
    predicted_labels = pipeline.predict(test_features)
    confusionMatrix = confusion_matrix(test_labels, predicted_labels)
    logging.info(f"Confusion Matrix: {confusionMatrix}")

    estimatorName = 'estimator_' + str(pipelineId) + '.pickle'
    logging.info(f"Estimator name: {estimatorName}")
    fullEstimatorName = estimatorStorePath + estimatorName
    filehandler = open(fullEstimatorName, "wb")
    estimatorContainer = dict()
    estimatorContainer["currencyEncoder"] = currencyEncoder
    estimatorContainer["countryEncoder"] = countryEncoder
    estimatorContainer["pipeline"] = pipeline
    pickle.dump(estimatorContainer, filehandler)
    filehandler.close()
    logging.info(f"Estimator: {estimatorName} pickled")
    persistEstmator(databaseName, estimatorName,estimatorContainer)

#     sendTrainMessage(pipelineId)
#
#
@app.route('/fit', methods=['POST'])
def fit():
    logging.info('Processing default request')
    availableFeatureSelectors = getFeatureSelectors()
    availableSamplers = getSamplers()
    availableScalers = getScalers()
    availableModels = getModels()
    ifTrainParametersAreProper = True
    # trainTaskId = request.get_json()
    trainTaskId = 1
    if trainTaskId is None:
        logging.error("trainTaskId isn't set")
        ifTrainParametersAreProper = False
    else:
        trainParameters = getTrainParameters(trainTaskId)
        databaseName = trainParameters[1]
        tableName = "transaction"
        if databaseName is None:
            logging.error("Train database isn't set")
            ifTrainParametersAreProper = False

        featureSelectorName = trainParameters[2]
        if featureSelectorName is None:
            logging.error("Feature selector isn't set")
            ifTrainParametersAreProper = False
        else:
            featureSelector = availableFeatureSelectors.get(featureSelectorName)
            if featureSelector is None:
                logging.error("Feature selector doesn't exist in train application")
                ifTrainParametersAreProper = False

        samplerName = trainParameters[3]
        if samplerName is None:
            logging.error("Sampler isn't set")
            ifTrainParametersAreProper = False
        else:
            sampler = availableSamplers.get(samplerName)
            if sampler is None:
                logging.error("Sampler doesn't exist in train application")
                ifTrainParametersAreProper = FalsesamplerName = trainParameters[3]

        scalerName = trainParameters[4]
        if scalerName is None:
            logging.error("Scaler isn't set")
            ifTrainParametersAreProper = False
        else:
            scaler = availableScalers.get(scalerName)
            if scaler is None:
                logging.error("Scaler doesn't exist in train application")
                ifTrainParametersAreProper = False

        modelName = trainParameters[5]
        if modelName is None:
            logging.error("Model isn't set")
            ifTrainParametersAreProper = False
        else:
            model = availableModels.get(modelName)
            if model is None:
                logging.error("Model doesn't exist in train application")
                ifTrainParametersAreProper = False

        expected_variance = trainParameters[7]
        if expected_variance is None:
            expected_variance = 0.99
            logging.info("Expected vatriance default value: 0.99")

        feature_engineering_switch = trainParameters[6]

        expectedVariance = trainParameters[7]

        if ifTrainParametersAreProper:
            databaseEncoder = encoding.DataBaseEncoder(logging)
            databaseEncoder.encode(databaseName)
            currencyEncoder=databaseEncoder.currencyEncoder
            countryEncoder=databaseEncoder.countryEncoder
            tableName = "encoded_transaction"
            if feature_engineering_switch == 1:
                logging.info("Feature engineering turned on")
                featureEngineer = engineering.FeatureEngineer(logging)
                featureEngineer.createNewFeatures(databaseName)
                tableName = "feature_engineered_transaction"
            configContainer = config.ConfigContainer()
            estimatorStorePath = configContainer.estimatorConfigDict.get("path")
            fitProcessParameter = dict()
            fitProcessParameter["trainTaskId"] = trainTaskId
            fitProcessParameter["databaseName"] = databaseName
            fitProcessParameter["tableName"] = tableName
            fitProcessParameter["featureSelector"] = featureSelector
            fitProcessParameter["expectedVariance"] = expectedVariance
            fitProcessParameter["sampler"] = sampler
            fitProcessParameter["scaler"] = scaler
            fitProcessParameter["model"] = model
            fitProcessParameter["estimatorStorePath"] = estimatorStorePath
            fitProcessParameter["currencyEncoder"] = currencyEncoder
            fitProcessParameter["countryEncoder"] = countryEncoder
            fitAndStoreEstimator(fitProcessParameter)
            response = "OK"
        else:
            response = "ERROR"
        logging.info(f'Response: {response}')
    return jsonify(response)


# from user import FeatureEngineering
# loggingAndConfigContainer=None
#
# def getLoggingAndConfigContainer():
#     if loggingAndConfigContainer is None:
#         loggingAndConfigContainer=configAndlogging.ConfigAndLoggingContainer()
#     return loggingAndConfigContainer


if __name__ == '__main__':
    print("Kezd")
    logging.basicConfig(filename='c:/Temp/Train.log', format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    # app.run(host='127.0.0.1', port=5000)
    app.run(port=8082)
    #     # sendTrainMessage("HELLO")
    # global configAndLoggingContainer

    # configAndLoggingContainer.estimatorConfigDict
    # with Manager() as manager:
    #     list = manager.list()
    # queue.put(fitProcessParameter)
    # p = Process(target=fitAndStoreEstimator, args=(queue,))
    # list.append("Teszt")
    # p = Process(target=f, args=(list,))
    # p.start()
    # p.join()
    # fit()
    logging.info("Train service started")
    print("END")
