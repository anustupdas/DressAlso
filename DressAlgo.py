import csv
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.matlib
import math

# from IPython.display import display, HTML


# TRAIN_PATH = 'dataset/csv_result-ship.csv'
TRAIN_PATH = 'dataset/Data_exp.csv'


def loadDatasetWithPandas(path):
    # Reading the raw data from csv file
    rawData = pd.read_csv(path)
    # display(rawData)
    # replacing the string indicating missing values with the numpy value for missing values
    # NaNProcessedData = rawData.replace({'na': np.nan}, regex=True)
    NaNProcessedData = rawData
    return NaNProcessedData


# This function is used to make subspaces, SelectedFeature will contain the previous best subspace.
def makeSubspaces(trainingData, selectedFeature):
    subspacesList = []
    if len(selectedFeature) == 0:
        for cols in trainingData.columns:
            # 'here we have included {} columns'.format(noOfFeature)
            subspacesList.append(cols)
    else:
        # subspacesList=[[cols,item] for cols in trainingData.columns for item in selectedFeature]
        for cols in trainingData.columns:
            # 'here we have included {} columns'.format(noOfFeature)
            subspacesList.append([cols, selectedFeature])

    return subspacesList


# This function generates a random score.
# Sumit_Respo
def calculateSubspaceScore(featureList):
    ConstrainedScore = random.randint(0, 100)
    DistanceScore = random.randint(0, 100)
    FinalScore = ConstrainedScore + DistanceScore
    return FinalScore


# This function will perform the DB-Scan Clustering Alorithm and compute the score on each subspaces.
# Input: Subspace of features.
# Output: Quality score for each subspace.
def performDBScan(subspace):
    featureSpace = []

    if type(subspace) == list:
        print("Selected Subscape")
        for i in subspace:
            if type(i) == list:
                for k in i:
                    # print(k)
                    featureSpace.append(k)
            else:
                # print(i)
                featureSpace.append(i)
    else:
        # print("Selected Subspace")
        # print(subspace)
        featureSpace.append(subspace)

    print("main")
    print(featureSpace)

    CandidateDataFrame = pd.DataFrame(trainData, columns=featureSpace)
    print("dataframe is: ")
    print(CandidateDataFrame)

    CreateDBCluster(CandidateDataFrame)

    FinalScore = calculateSubspaceScore(subspace)

    return FinalScore


# Implementation of DB Scan Algo
def CreateDBCluster(CandidateDataFrame):
    count = trainData.shape[0]

    D = math.log(count)
    minPts = int(D)
    kneighbour = minPts - 1

    data = trainData

    nbrs = NearestNeighbors(n_neighbors=minPts, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    sorted_distances = np.sort(distances, axis=None)

    d = distances[:, kneighbour]
    i = indices[:, 0]

    df = pd.DataFrame(data=d, columns=['values'])

    # converts the dataframe values to a list
    values = list(df['values'])

    # get length of the value set
    nPoints = len(values)
    allkdistpoints = np.vstack((range(nPoints), values)).T

    # Access the first and last point and plot a line between them
    largestkdistpoint = allkdistpoints[0]
    kdistlinevector = allkdistpoints[-1] - allkdistpoints[0]
    kdistlinevectorNorm = kdistlinevector / np.sqrt(np.sum(kdistlinevector ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vectorWithlargestkpoint = allkdistpoints - largestkdistpoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vectorWithlargestkpoint * np.matlib.repmat(kdistlinevectorNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, kdistlinevectorNorm)
    vecToLine = vectorWithlargestkpoint - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    maxdistance = np.amax(distToLine)
    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)

    # print "Knee of the curve is at index =",idxOfBestPoint
    # print "Knee value =", values[idxOfBestPoint]

    #print(maxdistance)
    #print(idxOfBestPoint)
    #print(values[idxOfBestPoint])

    # plot of the original curve and its corresponding distances
    plt.figure(figsize=(12, 6))
    plt.plot(distToLine, label='Distances', color='r')
    plt.plot(values, label='Series', color='b')
    plt.plot([idxOfBestPoint], values[idxOfBestPoint], marker='o', markersize=8, color="red", label='Knee')
    plt.legend()
    plt.show()

    epsilon = maxdistance

    ## FITING THE DATA WITH DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=minPts).fit(CandidateDataFrame)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # number of cluster ignoring noise point
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("No of Clusters")
    print(n_clusters_)


# Append a feature with the score. Example: [Rajesh , 150]
def scoreCalculate(ssList):
    subspaceScoresList = []
    # subArray = []
    for feature in ssList:
        # subArray.append(feature)
        FinalsubSpaceScore = performDBScan(feature)
        # subArray.clear()

        subspaceScoresList.append([feature, FinalsubSpaceScore])
    return subspaceScoresList


def bestScore(ssScoresList):
    return max_by_score(ssScoresList)


# Traverses the list to get the best score(Max score)
def max_by_score(sequence):
    if not sequence:
        raise ValueError('empty sequence')
    maximum = sequence[0]
    for item in sequence:
        if item[1] > maximum[1]:
            maximum = item
    return maximum


# This function iterates through the supspaces collects the best one and increases the cardinality every single time.
# df = Data, feat_select = Selected best feature, previousBestScore = Previous best feature according to the random score generated
# currentBestScore = Current best feature according to the random score genearted.

def iter_subspace(df, feat_select, previousBestScore, currentBestScore):
    while previousBestScore <= currentBestScore:
        possible_sslist = makeSubspaces(df, feat_select)
        score_sslist = scoreCalculate(possible_sslist)
        featureset_score = bestScore(score_sslist)
        features_for_nxt_iter = [item for item in df.columns if item not in featureset_score[0]]
        if featureset_score[0] not in features_selected:
            features_selected.append(featureset_score[0])
            previousBestScore = currentBestScore
            currentBestScore = featureset_score[1]
        featu = []
        inter_results = ConvertList(features_selected, featu)
        features_selected_final = getUniqueItems(featu)
        # print(possible_sslist)
        # print('')
        # print(score_sslist)
        # print('')
        print("Subspace With Best Score for the iteration")
        print(featureset_score)
        # print(features_for_nxt_iter)
        print('')
        print(features_selected_final)
        print("Previous Best Subspace Score")
        print(previousBestScore)
        print("Current Best Subspace Score")
        print(currentBestScore)
        # print('')
        # print('')
        iter_subspace(df[features_for_nxt_iter], features_selected_final, previousBestScore, currentBestScore)
        break


# This function is used to removed any duplicates in the list.

def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# This function is used to convert any two Dimensional List to a single List.#This f

def ConvertList(temp_list, featre):
    for ele in temp_list:
        if type(ele) == list:
            ConvertList(ele, featre)
        else:
            featre.append(ele)
    return featre


# Load Datasets
###############################################
# Load Train Dataset
trainData = loadDatasetWithPandas(TRAIN_PATH)
print(trainData)

# The main function of the program.
currentBestScore = 0
previousBestScore = 0
features_selected = []
iter_subspace(trainData, features_selected, previousBestScore, currentBestScore)




