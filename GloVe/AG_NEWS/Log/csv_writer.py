import csv
import os

global fileName

def setFileName(_fileName):
    global fileName

    fileName = _fileName

def getFileName():
    global fileName
    return fileName

def writeHeader(fields):
    if not os.path.exists(fileName):
        with open(fileName, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the header
            csvwriter.writerow(fields)

def writeRow(fields):
    with open(fileName, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the row
        csvwriter.writerow(fields)