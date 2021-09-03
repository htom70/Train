import logging
import math
import statistics
import multiprocessing as mp
from multiprocessing import Process
import numpy as np

from user import database

def getTransacTionAmountProperties(timestamp, amount, basePropertyCollection):
    amountCollectionByDateDictionary = dict()

    for currentTimestampAndAmount in basePropertyCollection:
        currentAmount = currentTimestampAndAmount[0]
        currentTimestamp = currentTimestampAndAmount[1]
        currentDate = int(currentTimestamp)
        if amountCollectionByDateDictionary.get(currentDate) is None:
            amountCollection = list()
            amountCollection.append(currentAmount)
            amountCollectionByDateDictionary[currentDate] = amountCollection
        else:
            amountCollection = amountCollectionByDateDictionary.get(currentDate)
            amountCollection.append(currentAmount)

    transactionDailyAmountCollectionSince3Days = list()
    transactionDailyAmountCollectionSince7Days = list()
    transactionDailyAmountCollectionSince15Days = list()
    transactionDailyAmountCollectionSince30Days = list()
    transactionDailyAmountCollectionSinceFirstDate = list()

    for date in amountCollectionByDateDictionary.keys():
        if date > timestamp - 3:
            transactionDailyAmountCollectionSince3Days.extend(amountCollectionByDateDictionary.get(date))
        if date > timestamp - 7:
            transactionDailyAmountCollectionSince7Days.extend(amountCollectionByDateDictionary.get(date))
        if date > timestamp - 15:
            transactionDailyAmountCollectionSince15Days.extend(amountCollectionByDateDictionary.get(date))
        if date > timestamp - 30:
            transactionDailyAmountCollectionSince30Days.extend(amountCollectionByDateDictionary.get(date))
        transactionDailyAmountCollectionSinceFirstDate.extend(amountCollectionByDateDictionary.get(date))

    averageAmountSince3Days = statistics.mean(transactionDailyAmountCollectionSince3Days) if len(
        transactionDailyAmountCollectionSince3Days) > 0 else 0
    deviationAmountSince3Days = statistics.stdev(transactionDailyAmountCollectionSince3Days) if len(
        transactionDailyAmountCollectionSince3Days) > 1 else 0
    medianAmountSince3Days = statistics.median(transactionDailyAmountCollectionSince3Days) if len(
        transactionDailyAmountCollectionSince3Days) > 0 else 0
    amountToAverageAmountRatioSince3Days = amount / averageAmountSince3Days if averageAmountSince3Days != 0 else 0
    amountToAverageAmountDiffSince3Days = amount - averageAmountSince3Days
    amountToMedianAmountRatioSince3Days = amount / medianAmountSince3Days if medianAmountSince3Days != 0 else 0
    amountToMedianAmountDiffSince3Days = amount - medianAmountSince3Days
    amountMinusAverageAmountToDeviationAmountRatioSince3Days = (
                                                                       amount - averageAmountSince3Days) / deviationAmountSince3Days if deviationAmountSince3Days != 0 else 0
    amountMinusAverageAmountToDeviationAmountDiffSince3Days = amount - averageAmountSince3Days - deviationAmountSince3Days

    averageAmountSince7Days = statistics.mean(transactionDailyAmountCollectionSince7Days) if len(
        transactionDailyAmountCollectionSince7Days) > 0 else 0
    deviationAmountSince7Days = statistics.stdev(transactionDailyAmountCollectionSince7Days) if len(
        transactionDailyAmountCollectionSince7Days) > 1 else 0
    medianAmountSince7Days = statistics.median(transactionDailyAmountCollectionSince7Days) if len(
        transactionDailyAmountCollectionSince7Days) > 0 else 0
    amountToAverageAmountRatioSince7Days = amount / averageAmountSince7Days if averageAmountSince7Days != 0 else 0
    amountToAverageAmountDiffSince7Days = amount - averageAmountSince7Days
    amountToMedianAmountRatioSince7Days = amount / medianAmountSince7Days if medianAmountSince7Days != 0 else 0
    amountToMedianAmountDiffSince7Days = amount - medianAmountSince7Days
    amountMinusAverageAmountToDeviationAmountRatioSince7Days = (
                                                                       amount - averageAmountSince7Days) / deviationAmountSince7Days if deviationAmountSince7Days != 0 else 0
    amountMinusAverageAmountToDeviationAmountDiffSince7Days = amount - averageAmountSince7Days - deviationAmountSince7Days

    averageAmountSince15Days = statistics.mean(transactionDailyAmountCollectionSince15Days) if len(
        transactionDailyAmountCollectionSince15Days) > 0 else 0
    deviationAmountSince15Days = statistics.stdev(transactionDailyAmountCollectionSince15Days) if len(
        transactionDailyAmountCollectionSince15Days) > 1 else 0
    medianAmountSince15Days = statistics.median(transactionDailyAmountCollectionSince15Days) if len(
        transactionDailyAmountCollectionSince15Days) > 0 else 0
    amountToAverageAmountRatioSince15Days = amount / averageAmountSince15Days if averageAmountSince15Days != 0 else 0
    amountToAverageAmountDiffSince15Days = amount - averageAmountSince15Days
    amountToMedianAmountRatioSince15Days = amount / medianAmountSince15Days if medianAmountSince15Days != 0 else 0
    amountToMedianAmountDiffSince15Days = amount - medianAmountSince15Days
    amountMinusAverageAmountToDeviationAmountRatioSince15Days = (
                                                                        amount - averageAmountSince15Days) / deviationAmountSince15Days if deviationAmountSince15Days != 0 else 0
    amountMinusAverageAmountToDeviationAmountDiffSince15Days = amount - averageAmountSince15Days - deviationAmountSince15Days

    averageAmountSince30Days = statistics.mean(transactionDailyAmountCollectionSince30Days) if len(
        transactionDailyAmountCollectionSince30Days) > 0 else 0
    deviationAmountSince30Days = statistics.stdev(transactionDailyAmountCollectionSince30Days) if len(
        transactionDailyAmountCollectionSince30Days) > 1 else 0
    medianAmountSince30Days = statistics.median(transactionDailyAmountCollectionSince30Days) if len(
        transactionDailyAmountCollectionSince30Days) > 0 else 0
    amountToAverageAmountRatioSince30Days = amount / averageAmountSince30Days if averageAmountSince30Days != 0 else 0
    amountToAverageAmountDiffSince30Days = amount - averageAmountSince30Days
    amountToMedianAmountRatioSince30Days = amount / medianAmountSince30Days if medianAmountSince30Days != 0 else 0
    amountToMedianAmountDiffSince30Days = amount - medianAmountSince30Days
    amountMinusAverageAmountToDeviationAmountRatioSince30Days = (
                                                                        amount - averageAmountSince30Days) / deviationAmountSince30Days if deviationAmountSince30Days != 0 else 0
    amountMinusAverageAmountToDeviationAmountDiffSince30Days = amount - averageAmountSince30Days - deviationAmountSince30Days

    averageAmountSinceFirstDay = statistics.mean(transactionDailyAmountCollectionSinceFirstDate) if len(
        transactionDailyAmountCollectionSinceFirstDate) > 0 else 0
    deviationAmountSinceFirstDay = statistics.stdev(transactionDailyAmountCollectionSinceFirstDate) if len(
        transactionDailyAmountCollectionSinceFirstDate) > 1 else 0
    medianAmountSinceFirstDay = statistics.median(transactionDailyAmountCollectionSinceFirstDate) if len(
        transactionDailyAmountCollectionSinceFirstDate) > 0 else 0
    amountToAverageAmountRatioSinceFirstDay = amount / averageAmountSinceFirstDay if averageAmountSinceFirstDay != 0 else 0
    amountToAverageAmountDiffSinceFirstDay = amount - averageAmountSinceFirstDay
    amountToMedianAmountRatioSinceFirstDay = amount / medianAmountSinceFirstDay if medianAmountSinceFirstDay != 0 else 0
    amountToMedianAmountDiffSinceFirstDay = amount - medianAmountSinceFirstDay
    amountMinusAverageAmountToDeviationAmountRatioSinceFirstDay = (
                                                                          amount - averageAmountSinceFirstDay) / deviationAmountSinceFirstDay if deviationAmountSinceFirstDay != 0 else 0
    amountMinusAverageAmountToDeviationAmountDiffSinceFirstDay = amount - averageAmountSinceFirstDay - deviationAmountSinceFirstDay

    resultDictionary = dict()
    resultDictionary["amountToAverageAmountRatioSince3Days"] = amountToAverageAmountRatioSince3Days
    resultDictionary["amountToAverageAmountDiffSince3Days"] = amountToAverageAmountDiffSince3Days
    resultDictionary["amountToMedianAmountRatioSince3Days"] = amountToMedianAmountRatioSince3Days
    resultDictionary["amountToMedianAmountDiffSince3Days"] = amountToMedianAmountDiffSince3Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountRatioSince3Days"] = amountMinusAverageAmountToDeviationAmountRatioSince3Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountDiffSince3Days"] = amountMinusAverageAmountToDeviationAmountDiffSince3Days

    resultDictionary["amountToAverageAmountRatioSince7Days"] = amountToAverageAmountRatioSince7Days
    resultDictionary["amountToAverageAmountDiffSince7Days"] = amountToAverageAmountDiffSince7Days
    resultDictionary["amountToMedianAmountRatioSince7Days"] = amountToMedianAmountRatioSince7Days
    resultDictionary["amountToMedianAmountDiffSince7Days"] = amountToMedianAmountDiffSince7Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountRatioSince7Days"] = amountMinusAverageAmountToDeviationAmountRatioSince7Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountDiffSince7Days"] = amountMinusAverageAmountToDeviationAmountDiffSince7Days

    resultDictionary["amountToAverageAmountRatioSince15Days"] = amountToAverageAmountRatioSince15Days
    resultDictionary["amountToAverageAmountDiffSince15Days"] = amountToAverageAmountDiffSince15Days
    resultDictionary["amountToMedianAmountRatioSince15Days"] = amountToMedianAmountRatioSince15Days
    resultDictionary["amountToMedianAmountDiffSince15Days"] = amountToMedianAmountDiffSince15Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountRatioSince15Days"] = amountMinusAverageAmountToDeviationAmountRatioSince15Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountDiffSince15Days"] = amountMinusAverageAmountToDeviationAmountDiffSince15Days

    resultDictionary["amountToAverageAmountRatioSince30Days"] = amountToAverageAmountRatioSince30Days
    resultDictionary["amountToAverageAmountDiffSince30Days"] = amountToAverageAmountDiffSince30Days
    resultDictionary["amountToMedianAmountRatioSince30Days"] = amountToMedianAmountRatioSince30Days
    resultDictionary["amountToMedianAmountDiffSince30Days"] = amountToMedianAmountDiffSince30Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountRatioSince30Days"] = amountMinusAverageAmountToDeviationAmountRatioSince30Days
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountDiffSince30Days"] = amountMinusAverageAmountToDeviationAmountDiffSince30Days

    resultDictionary["amountToAverageAmountRatioSinceFirstDay"] = amountToAverageAmountRatioSinceFirstDay
    resultDictionary["amountToAverageAmountDiffSinceFirstDay"] = amountToAverageAmountDiffSinceFirstDay
    resultDictionary["amountToMedianAmountRatioSinceFirstDay"] = amountToMedianAmountRatioSinceFirstDay
    resultDictionary["amountToMedianAmountDiffSinceFirstDay"] = amountToMedianAmountDiffSinceFirstDay
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountRatioSinceFirstDay"] = amountMinusAverageAmountToDeviationAmountRatioSinceFirstDay
    resultDictionary[
        "amountMinusAverageAmountToDeviationAmountDiffSinceFirstDay"] = amountMinusAverageAmountToDeviationAmountDiffSinceFirstDay
    return resultDictionary


def getTransacTionNumberProperties(timestamp, basePropertyCollection):
    timestampCollection = list()
    for item in basePropertyCollection:
        timestampCollection.append(item[1])

    transactionDailyNumberCollectionSince3Days = list()
    transactionDailyNumberCollectionSince7Days = list()
    transactionDailyNumberCollectionSince15Days = list()
    transactionDailyNumberCollectionSince30Days = list()
    transactionDailyNumberCollectionSinceFirstDay = list()

    transactionNumberByDateDictionary = dict()
    for currentTimeStamp in timestampCollection:
        currentDate = int(currentTimeStamp)
        if transactionNumberByDateDictionary.get(currentDate) is None:
            transactionNumberByDateDictionary[currentDate] = 1
        else:
            number = transactionNumberByDateDictionary.get(currentDate)
            number = number + 1
            transactionNumberByDateDictionary[currentDate] = number

    transactionNumberOnCurrentDay = transactionNumberByDateDictionary.get(int(timestamp))

    for date in transactionNumberByDateDictionary.keys():
        if date > timestamp - 3:
            transactionDailyNumberCollectionSince3Days.append(transactionNumberByDateDictionary.get(date))
        if date > timestamp - 7:
            transactionDailyNumberCollectionSince7Days.append(transactionNumberByDateDictionary.get(date))
        if date > timestamp - 15:
            transactionDailyNumberCollectionSince15Days.append(transactionNumberByDateDictionary.get(date))
        if date > timestamp - 30:
            transactionDailyNumberCollectionSince30Days.append(transactionNumberByDateDictionary.get(date))
        transactionDailyNumberCollectionSinceFirstDay.append(transactionNumberByDateDictionary.get(date))

    averageTransactionNumberSince3Days = statistics.mean(transactionDailyNumberCollectionSince3Days) if len(
        transactionDailyNumberCollectionSince3Days) > 0 else 0
    medianTransactionNumberSince3Days = statistics.median(transactionDailyNumberCollectionSince3Days) if len(
        transactionDailyNumberCollectionSince3Days) > 0 else 0
    transactionNumberToAverageDailyTransactionNumberRatioSince3Days = transactionNumberOnCurrentDay / averageTransactionNumberSince3Days if averageTransactionNumberSince3Days != 0 else 0
    transactionNumberToAverageDailyTransactionNumberDiffSince3Days = transactionNumberOnCurrentDay - averageTransactionNumberSince3Days
    transactionNumberToMedianDailyTransactionNumberRatioSince3Days = transactionNumberOnCurrentDay / medianTransactionNumberSince3Days if medianTransactionNumberSince3Days != 0 else 0
    transactionNumberToMedianDailyTransactionNumberDiffSince3Days = transactionNumberOnCurrentDay - medianTransactionNumberSince3Days

    averageTransactionNumberSince7Days = statistics.mean(transactionDailyNumberCollectionSince7Days) if len(
        transactionDailyNumberCollectionSince7Days) > 0 else 0
    medianTransactionNumberSince7Days = statistics.median(transactionDailyNumberCollectionSince7Days) if len(
        transactionDailyNumberCollectionSince7Days) > 0 else 0
    transactionNumberToAverageDailyTransactionNumberRatioSince7Days = transactionNumberOnCurrentDay / averageTransactionNumberSince7Days if averageTransactionNumberSince7Days != 0 else 0
    transactionNumberToAverageDailyTransactionNumberDiffSince7Days = transactionNumberOnCurrentDay - averageTransactionNumberSince7Days
    transactionNumberToMedianDailyTransactionNumberRatioSince7Days = transactionNumberOnCurrentDay / medianTransactionNumberSince7Days if medianTransactionNumberSince7Days != 0 else 0
    transactionNumberToMedianDailyTransactionNumberDiffSince7Days = transactionNumberOnCurrentDay - medianTransactionNumberSince7Days

    averageTransactionNumberSince15Days = statistics.mean(transactionDailyNumberCollectionSince15Days) if len(
        transactionDailyNumberCollectionSince15Days) > 0 else 0
    medianTransactionNumberSince15Days = statistics.median(transactionDailyNumberCollectionSince15Days) if len(
        transactionDailyNumberCollectionSince15Days) > 0 else 0
    transactionNumberToAverageDailyTransactionNumberRatioSince15Days = transactionNumberOnCurrentDay / averageTransactionNumberSince15Days if averageTransactionNumberSince15Days != 0 else 0
    transactionNumberToAverageDailyTransactionNumberDiffSince15Days = transactionNumberOnCurrentDay - averageTransactionNumberSince15Days
    transactionNumberToMedianDailyTransactionNumberRatioSince15Days = transactionNumberOnCurrentDay / medianTransactionNumberSince15Days if medianTransactionNumberSince15Days != 0 else 0
    transactionNumberToMedianDailyTransactionNumberDiffSince15Days = transactionNumberOnCurrentDay - medianTransactionNumberSince15Days

    averageTransactionNumberSince30Days = statistics.mean(transactionDailyNumberCollectionSince30Days) if len(
        transactionDailyNumberCollectionSince30Days) > 0 else 0
    medianTransactionNumberSince30Days = statistics.median(transactionDailyNumberCollectionSince30Days) if len(
        transactionDailyNumberCollectionSince30Days) > 0 else 0
    transactionNumberToAverageDailyTransactionNumberRatioSince30Days = transactionNumberOnCurrentDay / averageTransactionNumberSince30Days if averageTransactionNumberSince30Days != 0 else 0
    transactionNumberToAverageDailyTransactionNumberDiffSince30Days = transactionNumberOnCurrentDay - averageTransactionNumberSince30Days
    transactionNumberToMedianDailyTransactionNumberRatioSince30Days = transactionNumberOnCurrentDay / medianTransactionNumberSince30Days if medianTransactionNumberSince30Days != 0 else 0
    transactionNumberToMedianDailyTransactionNumberDiffSince30Days = transactionNumberOnCurrentDay - medianTransactionNumberSince30Days

    averageTransactionNumberSinceFirstDay = statistics.mean(transactionDailyNumberCollectionSinceFirstDay) if len(
        transactionDailyNumberCollectionSinceFirstDay) > 0 else 0
    medianTransactionNumberSinceFirstDay = statistics.median(transactionDailyNumberCollectionSinceFirstDay) if len(
        transactionDailyNumberCollectionSinceFirstDay) > 0 else 0
    transactionNumberToAverageDailyTransactionNumberRatioSinceFirstDay = transactionNumberOnCurrentDay / averageTransactionNumberSinceFirstDay if averageTransactionNumberSinceFirstDay != 0 else 0
    transactionNumberToAverageDailyTransactionNumberDiffSinceFirstDay = transactionNumberOnCurrentDay - averageTransactionNumberSinceFirstDay
    transactionNumberToMedianDailyTransactionNumberRatioSinceFirstDay = transactionNumberOnCurrentDay / medianTransactionNumberSinceFirstDay if medianTransactionNumberSinceFirstDay != 0 else 0
    transactionNumberToMedianDailyTransactionNumberDiffSinceFirstDay = transactionNumberOnCurrentDay - medianTransactionNumberSinceFirstDay

    resultDictionary = dict()
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberRatioSince3Days"] = transactionNumberToAverageDailyTransactionNumberRatioSince3Days
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberDiffSince3Days"] = transactionNumberToAverageDailyTransactionNumberDiffSince3Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberRatioSince3Days"] = transactionNumberToMedianDailyTransactionNumberRatioSince3Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberDiffSince3Days"] = transactionNumberToMedianDailyTransactionNumberDiffSince3Days

    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberRatioSince7Days"] = transactionNumberToAverageDailyTransactionNumberRatioSince7Days
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberDiffSince7Days"] = transactionNumberToAverageDailyTransactionNumberDiffSince7Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberRatioSince7Days"] = transactionNumberToMedianDailyTransactionNumberRatioSince7Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberDiffSince7Days"] = transactionNumberToMedianDailyTransactionNumberDiffSince7Days

    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberRatioSince15Days"] = transactionNumberToAverageDailyTransactionNumberRatioSince15Days
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberDiffSince15Days"] = transactionNumberToAverageDailyTransactionNumberDiffSince15Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberRatioSince15Days"] = transactionNumberToMedianDailyTransactionNumberRatioSince15Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberDiffSince15Days"] = transactionNumberToMedianDailyTransactionNumberDiffSince15Days

    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberRatioSince30Days"] = transactionNumberToAverageDailyTransactionNumberRatioSince30Days
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberDiffSince30Days"] = transactionNumberToAverageDailyTransactionNumberDiffSince30Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberRatioSince30Days"] = transactionNumberToMedianDailyTransactionNumberRatioSince30Days
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberDiffSince30Days"] = transactionNumberToMedianDailyTransactionNumberDiffSince30Days

    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberRatioSinceFirstDay"] = transactionNumberToAverageDailyTransactionNumberRatioSinceFirstDay
    resultDictionary[
        "transactionNumberToAverageDailyTransactionNumberDiffSinceFirstDay"] = transactionNumberToAverageDailyTransactionNumberDiffSinceFirstDay
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberRatioSinceFirstDay"] = transactionNumberToMedianDailyTransactionNumberRatioSinceFirstDay
    resultDictionary[
        "transactionNumberToMedianDailyTransactionNumberDiffSinceFirstDay"] = transactionNumberToMedianDailyTransactionNumberDiffSinceFirstDay
    return resultDictionary

def getTransacTionIntervalProperties(timestamp, basePropertyCollection):
    timestampCollection = list()
    for item in basePropertyCollection:
        timestampCollection.append(item[1])
    lengthOfTimestampCollection = len(timestampCollection)
    currentInterval = timestampCollection[0] - timestampCollection[1] if lengthOfTimestampCollection > 1 else 0
    intervalsSince3Days = list()
    intervalsSince7Days = list()
    intervalsSince15Days = list()
    intervalsSince30Days = list()
    intervalsSinceFirstDay = list()
    for i in range(0, lengthOfTimestampCollection - 1, 1):
        nextTimeStamp = timestampCollection[i + 1]
        currentTimeStamp = timestampCollection[i]
        interval = currentTimeStamp - nextTimeStamp
        if currentTimeStamp > timestamp - 3:
            intervalsSince3Days.append(interval)
        if currentTimeStamp > timestamp - 7:
            intervalsSince7Days.append(interval)
        if currentTimeStamp > timestamp - 15:
            intervalsSince15Days.append(interval)
        if currentTimeStamp > timestamp - 30:
            intervalsSince30Days.append(interval)
        intervalsSinceFirstDay.append(interval)

    averageIntervalSince3Days = statistics.mean(intervalsSince3Days) if len(intervalsSince3Days) > 0 else 0
    medianIntervalSince3Days = statistics.median(intervalsSince3Days) if len(intervalsSince3Days) > 0 else 0
    averageIntervalSince7Days = statistics.mean(intervalsSince7Days) if len(intervalsSince7Days) > 0 else 0
    medianIntervalSince7Days = statistics.median(intervalsSince7Days) if len(intervalsSince7Days) > 0 else 0
    averageIntervalSince15Days = statistics.mean(intervalsSince15Days) if len(intervalsSince15Days) > 0 else 0
    medianIntervalSince15Days = statistics.median(intervalsSince15Days) if len(intervalsSince15Days) > 0 else 0
    averageIntervalSince30Days = statistics.mean(intervalsSince30Days) if len(intervalsSince30Days) > 0 else 0
    medianIntervalSince30Days = statistics.median(intervalsSince30Days) if len(intervalsSince30Days) > 0 else 0
    averageIntervalSinceFirstDay = statistics.mean(intervalsSinceFirstDay) if len(intervalsSinceFirstDay) > 0 else 0
    medianIntervalSinceFirstDay = statistics.median(intervalsSinceFirstDay) if len(intervalsSinceFirstDay) > 0 else 0

    intervalToAverageIntervalRatioSince3Days = currentInterval / averageIntervalSince3Days if averageIntervalSince3Days != 0 else 0
    intervalToAverageIntervalDiffSince3Days = currentInterval - averageIntervalSince3Days
    intervalToMedianIntervalRatioSince3Days = currentInterval / medianIntervalSince3Days if medianIntervalSince3Days != 0 else 0
    intervalToMedianIntervalDiffSince3Days = currentInterval - medianIntervalSince3Days
    intervalToAverageIntervalRatioSince7Days = currentInterval / averageIntervalSince7Days if averageIntervalSince7Days != 0 else 0
    intervalToAverageIntervalDiffSince7Days = currentInterval - averageIntervalSince7Days
    intervalToMedianIntervalRatioSince7Days = currentInterval / medianIntervalSince7Days if medianIntervalSince7Days != 0 else 0
    intervalToMedianIntervalDiffSince7Days = currentInterval - medianIntervalSince7Days
    intervalToAverageIntervalRatioSince15Days = currentInterval / averageIntervalSince15Days if averageIntervalSince15Days != 0 else 0
    intervalToAverageIntervalDiffSince15Days = currentInterval - averageIntervalSince15Days
    intervalToMedianIntervalRatioSince15Days = currentInterval / medianIntervalSince15Days if medianIntervalSince15Days != 0 else 0
    intervalToMedianIntervalDiffSince15Days = currentInterval - medianIntervalSince15Days
    intervalToAverageIntervalRatioSince30Days = currentInterval / averageIntervalSince30Days if averageIntervalSince30Days != 0 else 0
    intervalToAverageIntervalDiffSince30Days = currentInterval - averageIntervalSince30Days
    intervalToMedianIntervalRatioSince30Days = currentInterval / medianIntervalSince30Days if medianIntervalSince30Days != 0 else 0
    intervalToMedianIntervalDiffSince30Days = currentInterval - medianIntervalSince30Days
    intervalToAverageIntervalRatioSinceFirstDay = currentInterval / averageIntervalSinceFirstDay if averageIntervalSinceFirstDay != 0 else 0
    intervalToAverageIntervalDiffSinceFirstDay = currentInterval - averageIntervalSinceFirstDay
    intervalToMedianIntervalRatioSinceFirstDay = currentInterval / medianIntervalSinceFirstDay if medianIntervalSinceFirstDay != 0 else 0
    intervalToMedianIntervalDiffSinceFirstDay = currentInterval - medianIntervalSinceFirstDay

    resultDictionary = dict()
    resultDictionary["intervalToAverageIntervalRatioSince3Days"] = intervalToAverageIntervalRatioSince3Days
    resultDictionary["intervalToAverageIntervalDiffSince3Days"] = intervalToAverageIntervalDiffSince3Days
    resultDictionary["intervalToMedianIntervalRatioSince3Days"] = intervalToMedianIntervalRatioSince3Days
    resultDictionary["intervalToMedianIntervalDiffSince3Days"] = intervalToMedianIntervalDiffSince3Days

    resultDictionary["intervalToAverageIntervalRatioSince7Days"] = intervalToAverageIntervalRatioSince7Days
    resultDictionary["intervalToAverageIntervalDiffSince7Days"] = intervalToAverageIntervalDiffSince7Days
    resultDictionary["intervalToMedianIntervalRatioSince7Days"] = intervalToMedianIntervalRatioSince7Days
    resultDictionary["intervalToMedianIntervalDiffSince7Days"] = intervalToMedianIntervalDiffSince7Days

    resultDictionary["intervalToAverageIntervalRatioSince15Days"] = intervalToAverageIntervalRatioSince15Days
    resultDictionary["intervalToAverageIntervalDiffSince15Days"] = intervalToAverageIntervalDiffSince15Days
    resultDictionary["intervalToMedianIntervalRatioSince15Days"] = intervalToMedianIntervalRatioSince15Days
    resultDictionary["intervalToMedianIntervalDiffSince15Days"] = intervalToMedianIntervalDiffSince15Days

    resultDictionary["intervalToAverageIntervalRatioSince30Days"] = intervalToAverageIntervalRatioSince30Days
    resultDictionary["intervalToAverageIntervalDiffSince30Days"] = intervalToAverageIntervalDiffSince30Days
    resultDictionary["intervalToMedianIntervalRatioSince30Days"] = intervalToMedianIntervalRatioSince30Days
    resultDictionary["intervalToMedianIntervalDiffSince30Days"] = intervalToMedianIntervalDiffSince30Days

    resultDictionary["intervalToAverageIntervalRatioSinceFirstDay"] = intervalToAverageIntervalRatioSinceFirstDay
    resultDictionary["intervalToAverageIntervalDiffSinceFirstDay"] = intervalToAverageIntervalDiffSinceFirstDay
    resultDictionary["intervalToMedianIntervalRatioSinceFirstDay"] = intervalToMedianIntervalRatioSinceFirstDay
    resultDictionary["intervalToMedianIntervalDiffSinceFirstDay"] = intervalToMedianIntervalDiffSinceFirstDay
    return resultDictionary

def createDatasetByCardDictionary(dataset):
    resultDictionary = dict()
    for record in dataset:
        cardNumber = record[1]
        if resultDictionary.get(cardNumber) is None:
            timestampCollection = list()
            timestampCollection.append(record)
            resultDictionary[cardNumber] = timestampCollection
        else:
            timestamps = resultDictionary.get(cardNumber)
            timestamps.append(record)
    return resultDictionary

def getTransacTionBaseNumberPropertiesUponDatesetDictionary(datasetByCardNumberDictionary, cardNumber, timestamp):
    resultCollection = list()
    records = datasetByCardNumberDictionary.get(cardNumber)
    for record in records:
        currentTimestamp = record[3]
        currentAmount = record[4]
        if currentTimestamp <= timestamp:
            resultCollection.append((currentAmount, currentTimestamp))
    return resultCollection

def processSubDataset(dataset, resultList):
    extendedDataset = list()
    transactionFeatures = dataset[:, 1:-1]
    tranactionLabels = dataset[:, -1:]
    length = len(transactionFeatures)
    for i in range(length):
        transactionFeature = transactionFeatures[i]
        currentCardNumber = math.floor(transactionFeature[0])
        currentTimestamp = transactionFeature[2]
        currentAmount = transactionFeature[3]
        transactionFeatureList = list(transactionFeature)
        # baseNumberProperties = getTransacTionBaseNumberProperties(databaseName, currentCardNumber, currentTimestamp) ADATBÁZIS lekérdezés helyett kódban leválogatva
        datasetByCardNumberDictionary = createDatasetByCardDictionary(dataset)
        baseNumberProperties = getTransacTionBaseNumberPropertiesUponDatesetDictionary(
            datasetByCardNumberDictionary, currentCardNumber, currentTimestamp)
        transacTionAmountProperties = getTransacTionAmountProperties(currentTimestamp, currentAmount,
                                                                     baseNumberProperties)
        transacTionNumberProperties = getTransacTionNumberProperties(currentTimestamp, baseNumberProperties)
        transacTionIntervalProperties = getTransacTionIntervalProperties(currentTimestamp, baseNumberProperties)
        keyAppends = ["3Days", "7Days", "15Days", "30Days", "FirstDay"]
        for keyAppend in keyAppends:
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountToAverageAmountRatioSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountToAverageAmountDiffSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountToMedianAmountRatioSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountToMedianAmountDiffSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountMinusAverageAmountToDeviationAmountRatioSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionAmountProperties.get(f"amountMinusAverageAmountToDeviationAmountDiffSince{keyAppend}"))

            transactionFeatureList.append(transacTionNumberProperties.get(
                f"transactionNumberToAverageDailyTransactionNumberRatioSince{keyAppend}"))
            transactionFeatureList.append(transacTionNumberProperties.get(
                f"transactionNumberToAverageDailyTransactionNumberDiffSince{keyAppend}"))
            transactionFeatureList.append(transacTionNumberProperties.get(
                f"transactionNumberToMedianDailyTransactionNumberRatioSince{keyAppend}"))
            transactionFeatureList.append(transacTionNumberProperties.get(
                f"transactionNumberToAverageDailyTransactionNumberDiffSince{keyAppend}"))

            transactionFeatureList.append(
                transacTionIntervalProperties.get(f"intervalToAverageIntervalRatioSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionIntervalProperties.get(f"intervalToAverageIntervalDiffSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionIntervalProperties.get(f"intervalToMedianIntervalRatioSince{keyAppend}"))
            transactionFeatureList.append(
                transacTionIntervalProperties.get(f"intervalToMedianIntervalDiffSince{keyAppend}"))
        label = tranactionLabels[i][0]
        transactionFeatureList.append(label)
        extendedDataset.append(transactionFeatureList)
    resultList.extend(extendedDataset)

def saveExtendedDataset(databaseName, extendedDataset):
    connection = database.getConnection()
    cursor = connection.cursor()
    cursor.execute("USE " + databaseName)
    insertScriptFile = open("SQL INSERT feature_engineered_transaction.txt", "r")
    sqlInsertScript = insertScriptFile.read()
    valuesArray = np.array(extendedDataset)
    bound = 1000
    length = len(extendedDataset)
    if length > bound:
        numberOfPartArray = int(length / bound)
        numberOfRestDatas = length - numberOfPartArray * bound
        for i in range(0, numberOfPartArray, 1):
            tempArray = valuesArray[i * bound:(i + 1) * bound, :]
            valueList = list()
            for record in tempArray:
                valueList.append(tuple(record))
            cursor.executemany(sqlInsertScript, valueList)
            connection.commit()
        tempArray = valuesArray[(numberOfPartArray) * bound:(numberOfPartArray) * bound + numberOfRestDatas, :]
        valueList = list()
        for record in tempArray:
            valueList.append(tuple(record))
        cursor.executemany(sqlInsertScript, valueList)
        connection.commit()
    else:
        valueList = list()
        for record in valuesArray:
            valueList.append(tuple(record))
        cursor.executemany(sqlInsertScript, valueList)
        connection.commit()
    cursor.close()
    connection.close()

# def createNewFeatures(databaseName):
#     dataSet= database.getAllRecordsFromDatabase(databaseName, tableName="encoded_transaction")
#     cpuCoreCount = mp.cpu_count()
#     print(f"Cpu logikai magok száma: {cpuCoreCount}")
#     lengthOfDataset = len(dataSet)
#     dataGroupCount = int(lengthOfDataset / cpuCoreCount)
#     dataGroupCollection = list()
#     for i in range(cpuCoreCount):
#         if i != cpuCoreCount - 1:
#             subDataset = dataSet[i * dataGroupCount:(i + 1) * dataGroupCount, :]
#             dataGroupCollection.append(subDataset)
#         else:
#             subDataset = dataSet[i * dataGroupCount:, :]
#             dataGroupCollection.append(subDataset)
#     processDictionary = dict()
#     resultFromProcessesDictionary = dict()
#     with mp.Manager() as manager:
#         for i in range(cpuCoreCount):
#             resultFromProcessesDictionary[i] = manager.list()
#             resultList = resultFromProcessesDictionary.get(i)
#             processDictionary[i] = Process(target=processSubDataset, args=(dataGroupCollection[i], resultList))
#         for i in range(cpuCoreCount):
#             processDictionary.get(i).start()
#         for i in range(cpuCoreCount):
#             processDictionary.get(i).join()
#         datasetFromProcesses = list()
#         for i in range(cpuCoreCount):
#             datasetFromProcesses.extend(resultFromProcessesDictionary.get(i))
#         print(len(datasetFromProcesses))
#     saveExtendedDataset(databaseName, datasetFromProcesses)

class FeatureEngineer():
    def __init__(self,logging):
        self.isEngineered = False
        self.logging=logging

    def createNewFeatures(self, databaseName):
        logging.info("create new features begin")
        tableName = "encoded_transaction"
        dataBaseHandler=database.DataBaseHandler(logging)
        dataSet = dataBaseHandler.getAllRecordsFromDatabase(databaseName, tableName)
        cpuCoreCount = mp.cpu_count()
        logging.info(f"CPU logical core number: {cpuCoreCount}")
        lengthOfDataset = len(dataSet)
        dataGroupCount = int(lengthOfDataset / cpuCoreCount)
        dataGroupCollection = list()
        for i in range(cpuCoreCount):
            if i != cpuCoreCount - 1:
                subDataset = dataSet[i * dataGroupCount:(i + 1) * dataGroupCount, :]
                dataGroupCollection.append(subDataset)
            else:
                subDataset = dataSet[i * dataGroupCount:, :]
                dataGroupCollection.append(subDataset)
        processDictionary = dict()
        resultFromProcessesDictionary = dict()
        with mp.Manager() as manager:
            for i in range(cpuCoreCount):
                resultFromProcessesDictionary[i] = manager.list()
                resultList = resultFromProcessesDictionary.get(i)
                processDictionary[i] = Process(target=processSubDataset, args=(dataGroupCollection[i], resultList))
            for i in range(cpuCoreCount):
                processDictionary.get(i).start()
            for i in range(cpuCoreCount):
                processDictionary.get(i).join()
            datasetFromProcesses = list()
            for i in range(cpuCoreCount):
                datasetFromProcesses.extend(resultFromProcessesDictionary.get(i))
            print(len(datasetFromProcesses))
        dataBaseHandler.createFeatureEngineeredTable(databaseName)
        saveExtendedDataset(databaseName, datasetFromProcesses)
        self.isEngineered=True
        logging.info("create new features end")
