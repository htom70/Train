import flask
from flask import request, jsonify
import json
import mysql.connector
import time
from multiprocessing import Process
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

app = flask.Flask(__name__)
app.config["DEBUG"] = True


def getConnection():
    connection = mysql.connector.connect(
        pool_name="local",
        pool_size=16,
        host="localhost",
        user="root",
        password="TOmi_1970")
    return connection


def getAllRecordsFromDatabase(databaseName):
    start = time.time()
    connection = getConnection()
    cursor = connection.cursor()
    sql_use_Query = "USE " + databaseName
    cursor.execute(sql_use_Query)
    sql_select_Query = "select * from transaction order by timestamp"
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
    connection = getConnection()
    cursor = connection.cursor()
    sqlUseQuery = "USE train_task"
    cursor.execute(sqlUseQuery)
    sqlSelectQuery = "select * from train_task where id = %s"
    parameter = (trainTaskId,)
    cursor.execute(sqlSelectQuery, parameter)
    result = cursor.fetchone()
    return result


def getSamplers():
    samplers = {
        'UnderSampler': RandomUnderSampler(sampling_strategy=0.5),
        'OverSampler': RandomOverSampler(sampling_strategy=0.5),
        'SMOTEENN': SMOTEENN(sampling_strategy=0.5, n_jobs=-1),
    }
    return samplers


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


def getFeatureSelectors():
    featureSelectors = {
        'RFE': RFECV(estimator=XGBClassifier(tree_method='gpu_hist', gpu_id=0), n_jobs=-1)
    }
    return featureSelectors


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


def sendTrainMessage(message):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='train', durable=True)
    message = {"trainId": 2 ,"result": "SUCCESFULLY"}
    channel.basic_publish(exchange='', routing_key='train', body=json.dumps(message))
    connection.close()


def storeAndRunFit(parameter):
    pipelineId = parameter.get('trainTaskId')
    databaseName = parameter.get("databaseName")
    sampler = parameter.get("sampler")
    scaler = parameter.get("scaler")
    featureSelector = parameter.get("featureSelector")
    model = parameter.get("model")

    dataset = getAllRecordsFromDatabase(databaseName)
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
    print(f"Confusion Matrix: {confusionMatrix}")
    pipelineName = 'pipeline_' + pipelineId + '.pickle'
    pathAndPipelineName = "C:\\Users\\machine\\Documents\\MKI\\Estimators\\Python\\" + pipelineName
    filehandler = open(pathAndPipelineName, "wb")
    pickle.dump(pipeline, filehandler)
    filehandler.close()
    sendTrainMessage('Train succesfully finished')


@app.route('/fit', methods=['POST'])
def fit():
    availableFeatureSelectors = getFeatureSelectors()
    availableSamplers = getSamplers()
    availableScalers = getScalers()
    availableModels = getModels()
    response = dict()
    content = request.get_json()
    trainTaskId = content.get("trainTaskId")
    if trainTaskId is None:
        response["trainTaskId"] = "trainTaskId doesn't exist"
    trainParameters = getTrainParameters(trainTaskId)
    if trainParameters is None:
        response["trainParameters"] = "Train Task doesn't exist"
    databaseName = trainParameters[1]
    if databaseName is None:
        response["databaseName"] = "Database doesn't exist"
    featureSelectorName = trainParameters[2]
    if featureSelectorName is not None:
        if availableFeatureSelectors.get(featureSelectorName) is not None:
            featureSelector = availableFeatureSelectors.get(featureSelectorName)
        else:
            response["featureSelectorName"] = "Feature selector is not installed in train application"
    else:
        response["featureSelectorName"] = "Feature selector is not specified"
    samplerName = trainParameters[3]
    if samplerName is not None:
        if availableSamplers.get(samplerName) is not None:
            sampler = availableSamplers.get(samplerName)
        else:
            response["samplerName"] = "Sampler is not installed in train application"
    else:
        response["samplerName"] = "Sampler is not specified"
    scalerName = trainParameters[4]
    if scalerName is not None:
        if availableScalers.get(scalerName) is not None:
            scaler = availableScalers.get(scalerName)
        else:
            response["scalerName"] = "Scaler is not installed in train application"
    else:
        response["scalerName"] = "Scaler is not specified"
    modelName = trainParameters[5]
    # if modelName is not None:
    #     if availableModels.get(modelName) is not None:
    #         model = availableModels.get(modelName)
    #     response["modelName"] = "Model is not installed in train application"
    # else:
    #     response["modelName"] = "Model is not specified"
    model = availableModels.get('Random Forest')
    if response.values() is not None:
        response["result"] = "Fit stored in train application"
    fitProcessParameter = dict()
    fitProcessParameter["trainTaskId"] = trainTaskId
    fitProcessParameter["databaseName"] = databaseName
    fitProcessParameter["featureSelector"] = featureSelector
    fitProcessParameter["sampler"] = sampler
    fitProcessParameter["scaler"] = scaler
    fitProcessParameter["model"] = model
    p = Process(target=storeAndRunFit, args=(fitProcessParameter,))
    p.start()
    return jsonify(response)


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000)
    # app.run(port=8083)
    sendTrainMessage("HELLO")
