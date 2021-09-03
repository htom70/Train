import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

from user import database


def convertCardNumberStringToInt(array):
    cardNumberStrings = array[:, 1:2]
    # cardNumberIntegers=cardNumberStrings.astype(np.long)
    print(type(cardNumberStrings[0][0]))
    cardNumberIntegers = np.array(cardNumberStrings, dtype=float)
    reshapedCardNumberIntegers = cardNumberIntegers.reshape(-1, 1)
    array[:, 1:2] = reshapedCardNumberIntegers


def convertVendorCodeStringToInt(array):
    vendorCodeStrings = array[:, 8:9]
    # vendorCodeIntegers=vendorCodeStrings.astype(np.int)
    vendorCodeIntegers = np.array(vendorCodeStrings, dtype=float)
    reshapedVendorCodeIntegers = vendorCodeIntegers.reshape(-1, 1)
    array[:, 8:9] = reshapedVendorCodeIntegers


def convertTimestampToJulian(array):
    convertedTimeStampDatas = list()
    for timeStamp in array[:, 3:4]:
        t = timeStamp[0]
        ts = pd.Timestamp(t)
        convertedTimeStampToJulian = ts.to_julian_date()
        convertedTimeStampDatas.append(convertedTimeStampToJulian)
    convertedTimeStampDataArray = np.array(convertedTimeStampDatas)
    reshaped_array = convertedTimeStampDataArray.reshape(-1, 1)
    array[:, 3:4] = reshaped_array


def convertCurrencyFeature(array):
    currenciesArray = array[:, 5:6]
    currencyEncoder = LabelEncoder()
    currencyEncoder.fit(currenciesArray)
    encodedCurrencies = currencyEncoder.transform(currenciesArray)
    reshapedEncodedCurrencies = encodedCurrencies.reshape(-1, 1)
    array[:, 5:6] = reshapedEncodedCurrencies
    return currencyEncoder


def convertCountryFeature(array):
    countries = array[:, 7:8]
    countryEncoder = LabelEncoder()
    modifiedCountries = column_or_1d(countries)
    countryEncoder.fit(modifiedCountries)
    encodedCountries = countryEncoder.transform(modifiedCountries)
    reshapedEncodedCountries = encodedCountries.reshape(-1, 1)
    array[:, 7:8] = reshapedEncodedCountries
    return countryEncoder


def saveData(dataBaseName, dataSet):
    connection = database.getConnection()
    valuesArray = dataSet[:, 1:]
    cursor = connection.cursor()
    sqlUseQuery = "USE " + dataBaseName
    cursor.execute(sqlUseQuery)
    # file = open("SQL INSERT feature_engineered_transaction.txt", "r")
    # sqlInsertQuery = file.read()
    sqlInsertQuery = "INSERt INTO encoded_transaction (card_number,transaction_type,timestamp,amount,currency_name,response_code,country_name,vendor_code,fraud) VALUES " \
                     "(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    length = len(valuesArray)
    bound = 1000
    if length > bound:
        numberOfPartArray = int(length / bound)
        numberOfRestDatas = length - numberOfPartArray * bound
        for i in range(0, numberOfPartArray, 1):
            tempArray = valuesArray[i * bound:(i + 1) * bound, :]
            valueList = list()
            for record in tempArray:
                valueList.append(tuple(record))
            cursor.executemany(sqlInsertQuery, valueList)
            connection.commit()
        tempArray = valuesArray[(numberOfPartArray) * bound:(numberOfPartArray) * bound + numberOfRestDatas, :]
        valueList = list()
        for record in tempArray:
            valueList.append(tuple(record))
        cursor.executemany(sqlInsertQuery, valueList)
        connection.commit()
    else:
        valueList = list()
        for record in valuesArray:
            valueList.append(tuple(record))
        cursor.executemany(sqlInsertQuery, valueList)
        connection.commit()


# def encode(databaseName):
#     rawDataSet = database.getAllRawRecordsFromDatabase(databaseName)
#     encodedTableName = "encoded_transaction"
#     database.createEncodedTable(databaseName, encodedTableName)
#     convertTimestampToJulian(rawDataSet)
#     convertCardNumberStringToInt(rawDataSet)
#     convertVendorCodeStringToInt(rawDataSet)
#     countryEncoder=convertCountryFeature(rawDataSet)
#     currencyEncoder=convertCurrencyFeature(rawDataSet)
#     saveData(databaseName, rawDataSet)
#     logging("Raw database encoded")


def isTransactionTableEncoded(dataBaseName):
    result = False
    connection = database.getConnection()
    cursor = connection.cursor()
    sqlUseQuery = "USE " + dataBaseName
    cursor.execute(sqlUseQuery)
    sqlEncodedCountQuery = "SELECT COUNT(*) FROM encoded_transaction"
    cursor.execute(sqlEncodedCountQuery)
    numOfEncodedTransactions = cursor.fetchone()
    sqlCountQuery = "SELECT COUNT(*) FROM transaction"
    cursor.execute(sqlCountQuery)
    numOfTransactions = cursor.fetchone()
    if (numOfEncodedTransactions == numOfTransactions):
        result = True
    return result


class DataBaseEncoder:
    def __init__(self,logging):
        self.isEncoded = False
        self.currencyEncoder = None
        self.countryEncoder = None
        self.logging=logging

    def encode(self, databaseName):
        databaseHandler=database.DataBaseHandler(logging)
        tableName = "transaction"
        rawDataSet = databaseHandler.getAllRecordsFromDatabase(databaseName, tableName)
        databaseHandler.createEncodedTable(databaseName)
        convertTimestampToJulian(rawDataSet)
        convertCardNumberStringToInt(rawDataSet)
        convertVendorCodeStringToInt(rawDataSet)
        self.countryEncoder = convertCountryFeature(rawDataSet)
        self.currencyEncoder = convertCurrencyFeature(rawDataSet)
        self.isEncoded=True
        saveData(databaseName, rawDataSet)
        logging.info("Transaction encoded")
        print("Kész")



