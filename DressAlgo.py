import csv
from pathlib import Path
import pandas as pd
import numpy as np

#from IPython.display import display, HTML


TRAIN_PATH = 'dataset/csv_result-ship.csv'




def loadDatasetWithPandas(path):
    # Reading the raw data from csv file
    rawData = pd.read_csv(path)
    #display(rawData)
    # replacing the string indicating missing values with the numpy value for missing values
    NaNProcessedData = rawData.replace({'na': np.nan}, regex=True)
    return NaNProcessedData


#This function will create n subspaces for each iteration staring from subspaces with 1 feature to n features.
#Input: Training Dataframe, No of features that each subspace shall contain.
#Output: A list containing all the possible subspaces for that itteration of n features.
def makeSubspaces(trainingData,noOfFeature):

    return subspacesList


#This function will perform the DB-Scan Clustering Alorithm and compute the score on each subspaces.
#Input: Subspace of features.
#Output: Quality score for each subspace.
def performDBScan(subspace):

    #Place to implement the DB scan Algo with the subscapce in hand.

    Score = calculateSubspaceScore(subspace)

    return Score



#This function will calculate the quality score of each subspace based on the clustering performed by DB scan.
#Input:
#Output: Quality score for each subspace.
def calculateSubspaceScore():

    ConstrainedScore = 0
    DistanceScore = 0
    FinalScore = ConstrainedScore + DistanceScore

    return FinalScore


#This function will check the best score from the subspacescorelist .
#Input: subSpaceScoreList ---> [subSpaceeName[i], QualityScore[i]]
#Output: Best[subSpaceeName, QualityScore].
def getBestSubspace(subSpaceScoreList):

    bestSubspace = [[],[]]

    #Check the list with highest qualityscore and append the name and score of the subspace in that list.

    return bestSubspace


#This function iterate over all the subspaces of lenght n and perform clustering and calculate qualityscore for each Subspace.
#Input: subSpaceList ---> [subSpaceeName]
#Output: Best[subSpaceeName, QualityScore].
def createCluster(subspacesList):

    subSpaceScoreList = []

    for i in subspacesList:
        subSpaceScore = performDBScan(i)
        subSpaceScoreList.append(subSpaceScore,i)


    bestSubspace = getBestSubspace(subSpaceScoreList)
    return bestSubspace











# Load Datasets
###############################################
# Load Train Dataset
trainData = loadDatasetWithPandas(TRAIN_PATH)
print(trainData)

noOfFeature = 1;
currentBestScore = 0;
previousBestScore = 0;
subspacesList = [];
while(currentBestScore >= previousBestScore):
    if noOfFeature != 1:
        previousBestScore = currentBestScore

    subspacesList =  makeSubspaces(trainData,noOfFeature)
    bestSubspace = createCluster(subspacesList)
    currentBestScore = bestSubspace[0][1]
    noOfFeature = noOfFeature + 1
else:
    #End the flow with the best subspace.
    print(bestSubspace)





