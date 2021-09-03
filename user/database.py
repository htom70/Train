import time
import numpy as np
import mysql.connector
import logging

# from user.ConfigAndLoggingProvider import ConfigAndLoggingContainer


class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)

def getConnection():
    connection = mysql.connector.connect(
        pool_name="local",
        pool_size=16,
        host="localhost",
        user="root",
        password="TOmi_1970")
    connection.set_converter_class(NumpyMySQLConverter)
    return connection

# def getAllRecordsFromDatabase(databaseName, tableName):
#     start = time.time()
#     connection = getConnection()
#     cursor = connection.cursor()
#     sql_use_Query = "USE " + databaseName
#     cursor.execute(sql_use_Query)
#     sql_select_Query = f"select * from {tableName} group by card_number, timestamp order by card_number, timestamp desc"
#     cursor.execute(sql_select_Query)
#     result = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     numpy_array = np.array(result)
#     end = time.time()
#     elapsedTime = end - start
#     logging.info(f'{databaseName:} {tableName} adatok beolvasva, betöltési idő: {elapsedTime}, rekordszám: {numpy_array.shape}')
#     return numpy_array[:, :]


# def createEncodedTable(databaseName):
#     connection = getConnection()
#     cursor = connection.cursor()
#     cursor.execute("USE " + databaseName)
#     file = open("SQL DROP TABLE encoded_transaction.txt", "r")
#     sqlDropTableScript = file.read()
#     cursor.execute(sqlDropTableScript)
#     connection.commit()
#     file = open("SQL CREATE TABLE encoded_transaction.txt", "r")
#     sqlCreataTableScript = file.read()
#     cursor.execute(sqlCreataTableScript)
#     connection.commit()
#     connection.close()
#     logging.info("encoded table created")

# def createFeatureEngineeredTable(databaseName):
#     connection = getConnection()
#     cursor = connection.cursor()
#     cursor.execute("USE " + databaseName)
#     file = open("SQL DROP TABLE feature_engineered_transaction.txt", "r")
#     sqlDropTableScript = file.read()
#     cursor.execute(sqlDropTableScript)
#     connection.commit()
#     file = open("SQL CREATE TABLE feature_engineered_transaction.txt", "r")
#     sqlCreataTableScript = file.read()
#     cursor.execute(sqlCreataTableScript)
#     connection.commit()
#     connection.close()
#     logging.info("feature engineered table created")

class DataBaseHandler():
    def __init__(self,logging):
        self.logging=logging

    def getAllRecordsFromDatabase(self,databaseName, tableName):
        start = time.time()
        connection = getConnection()
        cursor = connection.cursor()
        sql_use_Query = "USE " + databaseName
        cursor.execute(sql_use_Query)
        sql_select_Query = f"select * from {tableName} group by card_number, timestamp order by card_number, timestamp desc"
        cursor.execute(sql_select_Query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        numpy_array = np.array(result)
        end = time.time()
        elapsedTime = end - start
        logging.info(
            f'database: {databaseName}, table: {tableName} loaded, loading time: {elapsedTime}, record number: {numpy_array.shape}')
        return numpy_array[:, :]

    def createEncodedTable(self,databaseName):
        connection = getConnection()
        cursor = connection.cursor()
        cursor.execute("USE " + databaseName)
        file = open("SQL DROP TABLE encoded_transaction.txt", "r")
        sqlDropTableScript = file.read()
        cursor.execute(sqlDropTableScript)
        connection.commit()
        file = open("SQL CREATE TABLE encoded_transaction.txt", "r")
        sqlCreataTableScript = file.read()
        cursor.execute(sqlCreataTableScript)
        connection.commit()
        connection.close()
        logging.info("encoded table created")

    def createFeatureEngineeredTable(self,databaseName):
        connection = getConnection()
        cursor = connection.cursor()
        cursor.execute("USE " + databaseName)
        file = open("SQL DROP TABLE feature_engineered_transaction.txt", "r")
        sqlDropTableScript = file.read()
        cursor.execute(sqlDropTableScript)
        connection.commit()
        file = open("SQL CREATE TABLE feature_engineered_transaction.txt", "r")
        sqlCreataTableScript = file.read()
        cursor.execute(sqlCreataTableScript)
        connection.commit()
        connection.close()
        logging.info("feature engineered table created")

