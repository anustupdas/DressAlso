import csv
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.matlib
import math as math
import itertools
import sys


isCatagorical = False
catagoricalFtrPosList = []
NominalFtrPosList = []

## List of all ML constraints belonging to class - pos
posMLCons = []

## List of all ML constraints belonging to class - neg
negMLCons = []

## Create a list of NL constraints belonging to class - pos and neg
posNegNLCons = []

## List of pairs of ML constraints (randomly selected)
listMLConsPairs = []

## List of pairs of NL constraints (randomly selected)
listNLConsPairs = []


## Generate constraint pairs
def createRandomConsPairs(listCons, n):
    ## Generate all possible non-repeating pairs
    pairsCons = list(itertools.combinations(listCons, 2))

    ## Randomly shuffle these pairs
    random.shuffle(pairsCons)

    ## Randomly pick and return required no of pairs
    return random.sample(pairsCons, n)


## Create a list of all ML constraints
def createMLConsList():
    for index, row in dataRaw.iterrows():
        if float(row["mrt_liverfat_s2"]) <= 10:
            negMLCons.append(index)
            # unlabDataInst.append(row)
        elif float(row["mrt_liverfat_s2"]) > 10:
            posMLCons.append(index)
            # unlabDataInst.append(row)
        # else:
        # unlabDataInst.append(index)

    return createRandomConsPairs(posMLCons, int(noMLCons / 2)) + createRandomConsPairs(negMLCons, int(noMLCons / 2))


## Create a list of (pairs of) all NL constraints
def createNLConsList():
    for i, j in zip(posMLCons, negMLCons):
        tupNLCons = (i, j)
        posNegNLCons.append(tupNLCons)

    ## Create a list of (pairs of) NL constraints
    return random.sample(posNegNLCons, noNLCons)


## Check whether the feature is nominal or not
def checkNominal(feature):
    nominal = True

    return nominal


## Calculate the distance between object pairs in a feature
def calculateSqDistDiff(feature, objPairX, objPairY):
    ## If both oject pairs are nominal
    # if (checkNominal(feature)):
    #    if (objPairX == objPairY):
    #        diffDist = 0
    #    else:
    #        diffDist = 1
    ## If both object pairs are continuous
    # else:
    diffDist = objPairX - objPairY
    # print("Feature:",  feature)
    # print("objPairX = ", type(objPairX))
    # print("objPairY = ", objPairY)

    return (diffDist ** 2)


## Calculate distance between object pairs for every feature in a feature space using Heterogeneous Euclidean Overlap Metric
def calculateHEOM(subspace, objPairX, objPairY):
    sumDistSq = 0

    ## Calculate distance between object pairs for every feature in a feature space using Heterogeneous Euclidean Overlap Metric
    for feature in subspace.columns:
        sumDistSq = sumDistSq + calculateSqDistDiff(feature, subspace.iloc[objPairX][feature],
                                                    subspace.iloc[objPairY][feature])

    return math.sqrt(sumDistSq)


## Calculate the average distance between object pairs in a subspace
def calculateAvgDist(subspace, listConsPairs):
    totalDist = 0

    for consPair in listConsPairs:
        totalDist = calculateHEOM(subspace, consPair[0], consPair[1])

    ## Total no of ML/NL constraints
    noConst = len(listMLConsPairs) + len(listNLConsPairs)

    avgDist = totalDist / noConst

    return avgDist


## Calculate the quality score of a subspace based on distance
def calculateDistScore(subspace):
    ## Average distance between ML objects pairs
    avgDistML = calculateAvgDist(subspace, listMLConsPairs)

    ## Average distance between NL objects pairs
    avgDistNL = calculateAvgDist(subspace, listNLConsPairs)

    ## Quality score based on distance
    qualScoreDist = avgDistNL - avgDistML

    return qualScoreDist


## Calculate the total no of satisfied NL constraints
def calculateNoSatisNLCons():
    i = 0

    listClusterCons = []

    #listClusterNLCons = []
    listCommCluster = []
    for constPair in listNLConsPairs:

        listCommCluster.clear()
        listClusterCons.clear()
        #listClusterNLCons.clear()

        for clusterNo in range(len(position_list)):

            #print('elements in cluster: ', position_list[clusterNo])
            if constPair[0] in position_list[clusterNo]:
                listClusterCons.append(clusterNo)
            #print('listClusterMLCons: ', listClusterCons)
            if constPair[1] in position_list[clusterNo]:
                listClusterCons.append(clusterNo)
            #print('listClusterNLCons: ', listClusterCons)

        #listCommCluster = list(set(listClusterMLCons).symmetric_difference(set(listClusterNLCons)))
        listCommCluster = list(set(listClusterCons))
        #print('listCommCluster: ',listCommCluster)
        if len(listCommCluster) == 2:
            i = i + 1

    return i


## Calculate the total no of satisfied ML constraints
def calculateNoSatisMLCons():
    i = 0

    for constPair in listMLConsPairs:
        for clusterNo in range(len(position_list)):
            if constPair[0] in position_list[clusterNo] and constPair[1] in position_list[clusterNo]:
                i = i + 1

    return i


## Calculate the quality score of a subspace based on constraint satisfaction
def calculateConstScore():
    ## No of satisfied ML constraints
    noSatisML = calculateNoSatisMLCons()
    #print('No of satis ML: ', noSatisML)
    ## No of satisfied NL constraints
    noSatisNL = calculateNoSatisNLCons()
    #print('No of satis NL: ', noSatisNL)

    ## Total no of ML constraints
    totalNoML = len(listMLConsPairs)
    #print('Total No of ML: ', totalNoML)
    ## Total no of NL constraints
    totalNoNL = len(listNLConsPairs)
    #print('Total No of NL: ', totalNoNL)
    ## Quality score based on constraint satisfaction
    qualScoreConst = (noSatisML + noSatisNL) / (totalNoML + totalNoNL)
    return qualScoreConst


## Calculate the quality score of each subspace based on the clustering performed by DBSCAN algorithm.
# Input:
# Output: Quality score for each subspace.
def calculateSubspaceScore(subspace):
    ## Quality score based on constraint satisfaction
    constraintScore = calculateConstScore()

    ## Quality score based on distance
    distanceScore = calculateDistScore(subspace)

    if distanceScore < 0:
        negDistSubspace.append(subspace.head(0))
    ## Final quality score
    finalScore = constraintScore + distanceScore

    return finalScore


# from IPython.display import display, HTML
## Test Data
# TRAIN_PATH = '/media/sumit/Entertainment/OVGU - DKE/Summer 2018/Medical Data Mining/csv_result-ship_14072018.csv'

## Original Data
# TRAIN_PATH = '/media/sumit/Entertainment/OVGU - DKE/Summer 2018/Medical Data Mining/csv_result-ship_22042018.csv'

## Labeled Data
TRAIN_PATH = 'dataset/csv_result-ship_14072018.csv'
#TRAIN_PATH ='dataset/csv_result-ship_labeled_data.csv'


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
# def calculateSubspaceScore(featureList):
#    ConstrainedScore = random.randint(0, 100)
#    DistanceScore = random.randint(0, 100)
#    FinalScore = ConstrainedScore + DistanceScore
#    return FinalScore


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

    distScoreSubspace = calculateDistScore(CandidateDataFrame)

    print("Distance Score: ", distScoreSubspace)

    if distScoreSubspace < 0:
        negDistSubspace.append(CandidateDataFrame.columns)
        totalQualScoreSubspace = distScoreSubspace
        print("Distance Score: ", totalQualScoreSubspace)
        return totalQualScoreSubspace

    print("calling DB scan")
    constScoreSubspace = CreateDBCluster(CandidateDataFrame)

    totalQualScoreSubspace = distScoreSubspace + constScoreSubspace

    return totalQualScoreSubspace


position_list = []

def mydistance(x,y):
    catDist = 0
    nomDist = 0
    totDist = 0
    if isCatagorical:
        for i in catagoricalFtrPosList:
            if x[i] == y[i]:
                dist = 0
            else:
                dist = 1
            catDist = catDist + dist

    if len(NominalFtrPosList) > 0:
        for j in NominalFtrPosList:
            Ndist = ((x[j]-y[j])**2)
            nomDist = nomDist + Ndist

    totDist = catDist + nomDist
    #print("DBSCAN Dist: ", totDist)
    return totDist
 # print("Value of X", len(x))
  #print("Value of Y", len(y))
 # print("Value of X", x)
 # print("Last Value of X", x[len(x) - 1])
 # print("Value of Y", y)
 # print("Last Value of Y", y[len(y) - 1])
  #return numpy.sum((x-y)**2)


# Implementation of DB Scan Algo
def CreateDBCluster(CandidateDataFrame):
    position_list.clear()



    print("Catagorical Features: ")
    print(catagoricalFeature)

    currentSubspaceFtre = CandidateDataFrame.columns
    print("Currect Subspace Colums are")
    print(currentSubspaceFtre)

    catagoricalFtrPosList.clear()
    NominalFtrPosList.clear()
    isCatagorical = False
    count = 0
    for ftr in currentSubspaceFtre:
        if ftr in catagoricalFeature:
            isCatagorical = True
            catagoricalFtrPosList.append(count)
        else:
            NominalFtrPosList.append(count)
        count = count + 1

    print("catagoricalFtrPosList: ", catagoricalFtrPosList)
    print("NominalFtrPosList: ", NominalFtrPosList)
    print("isCatagorical?? ", isCatagorical)
    ## FITING THE DATA WITH DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=minPts, metric=mydistance).fit(CandidateDataFrame)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    lable_list = []

    print("Labels are: ")
    a = 0;
    for i in labels:
        lable_list.append(i)
    # print(lable_list)

    output = []
    for x in lable_list:
        if x not in output:
            output.append(x)
    print("Unique values")
    print(output)

    # position_list = []
    for k in output:
        if k != -1:
            a = 0
            custer_list = []
            for l in lable_list:

                if l == k:
                    custer_list.append(a)
                a = a + 1

            position_list.append(custer_list)

    # print("Position List")
    # print(position_list)

    # number of cluster ignoring noise point
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("No of Clusters")
    print(n_clusters_)

    # FinalScore = calculateSubspaceScore(CandidateDataFrame)
    constScore = calculateConstScore()
    print("Constraint Score: ", constScore)
    return constScore


# Append a feature with the score. Example: [Rajesh , 150]
def scoreCalculate(ssList):
    print("SSList:", ssList)
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


##Filter Addition
def dropNegetiveScoreFeature(ssScoresList):
    negFeatureList = []
    print("Score List: ", ssScoresList)
    print("Size of Score List: ", len(ssScoresList))
    for i in ssScoresList:
        if i[1] < 0:
            negFeatureList.append(i)
            print("Negetive Score", i)

    return negFeatureList


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

        ##Filter Addition
        negFeatures = dropNegetiveScoreFeature(score_sslist)

        features_for_nxt_iter = [item for item in df.columns if item not in featureset_score[0]]

        ##Filter Addition
        print("No of neg subspace", len(negFeatures))
        if len(negFeatures) != 0:
            for Negitem in negFeatures:
                if Negitem[0] in features_for_nxt_iter:
                    features_for_nxt_iter.remove(Negitem[0])

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
        print("features_for_nxt_iter")
        print(features_for_nxt_iter)
        print('')
        print(features_selected_final)
        print("Size of features_for_nxt_iter: ", len(features_for_nxt_iter))
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


# This function is used to convert any two Dimensional List to a single List.
def ConvertList(temp_list, featre):
    for ele in temp_list:
        if type(ele) == list:
            ConvertList(ele, featre)
        else:
            featre.append(ele)
    return featre


## Calculate Min Points
def calcMinPts():
    count = trainData.shape[0]

    D = math.log(count)
    minPts = int(D)
    return minPts


def calcEpsilon():
    data = trainData

    minPts = calcMinPts()
    kneighbour = minPts - 1
    nbrs = NearestNeighbors(n_neighbors=minPts, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    d = distances[:, kneighbour]
    # i = indices[:, 0]
    sorted_distances = np.sort(d, axis=None)
    df = pd.DataFrame(data=sorted_distances, columns=['values'])

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

    scalarProduct = np.sum(vectorWithlargestkpoint * np.matlib.repmat(kdistlinevectorNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, kdistlinevectorNorm)
    vecToLine = vectorWithlargestkpoint - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    maxdistance = np.amax(distToLine)
    # knee/elbow is the point with max distance value
    # idxOfBestPoint = np.argmax(distToLine)

    return maxdistance


## Load the entire dataset into a data frame
dataRaw = loadDatasetWithPandas(TRAIN_PATH)

## User sets the no of ML constraints
# noMLCons = input("Enter the number of must-link constraints to be used:")
noMLCons = 10
## User sets the no of NL constraints
# noNLCons = input("Enter the number of not-link constraints to be used:")
noNLCons = 10

## List of randomly selected ML constraint pairs
#listMLConsPairs = createMLConsList()
# listMLConsPairs = [(126, 160), (21, 503), (238, 433), (127, 521), (422, 512), (35, 212), (212, 267), (391, 396), (71, 252), (4, 465)]
listMLConsPairs = [(69, 422), (469, 561), (144, 261), (505, 569), (109, 176), (304, 385), (111, 480), (196, 387), (331, 491), (101, 447)]

## List of randomly selected NL constraint pairs
#listNLConsPairs = createNLConsList()
# listNLConsPairs = [(393, 117), (28, 6), (219, 88), (21, 2), (41, 12), (13, 1), (239, 95), (207, 85), (155, 68), (134, 53)]
listNLConsPairs = [(406, 128), (232, 93), (223, 91), (413, 129), (206, 84), (218, 87), (563, 200), (150, 63), (545, 196), (47, 16)]

## Replace '?' with 'NaN'
dataRaw = dataRaw.replace('?', np.NaN)

k = dataRaw.nunique()
j = pd.unique(dataRaw.columns.values)

uniqueValFtreList = []
catagoricalFeature = []

a = 0;
b = 0;
for i in k:
    b = 0
    for l in j:
        if a == b:
            uniqueValFtreList.append([l, i])
        b = b+1
    a = a + 1

print("****************Feature with possible Categorial Data********************")
c = 0
for x in uniqueValFtreList:
    if x[1] <=20:
        catagoricalFeature.append(x[0])
        print(x)
        c= c+1

print("Lenght", c)





## Delete the unwanted features such as ones have date, time, id and class label stoed in it
trainData = dataRaw[dataRaw.columns.difference(
    ['id', 'exdate_ship_s0', 'exdate_ship_s1', 'exdate_ship_s2', 'exdate_ship0_s0', 'blt_beg_s0', 'blt_beg_s1',
     'blt_beg_s2', 'mrt_liverfat_s2'])]
print(trainData)

negDistSubspace = []
# trainData = pd.DataFrame([])

# for index, row in trainDataRaw.iterrows():
#    if str(row["mrt_liverfat_s2"]) != "nan":
#        trainData = trainData.append(row)

# unlabDataInst = pd.DataFrame([])

##Here
# import sys
# sys.exit("Stop")


for data in trainData.columns:
    if trainData[data].dtype == 'O':
        unique_elements = trainData[data].unique().tolist()

        trainData[data] = trainData[data].apply(lambda x: unique_elements.index(x))

# trainData.drop(columns = ['id', 'mrt_liverfat_s2'])
# trainData = trainDataRaw[trainDataRaw.columns.difference(['id', 'mrt_liverfat_s2'])]

epsilon = calcEpsilon()

minPts = calcMinPts()

# The main function of the program.
currentBestScore = 0
previousBestScore = 0
features_selected = []
iter_subspace(trainData, features_selected, previousBestScore, currentBestScore)