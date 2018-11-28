from sklearn.model_selection import *
from sklearn.neighbors import *
from sklearn.metrics import *
from sklearn.cluster import DBSCAN
from sklearn import preprocessing as pp
from numpy import array
from sklearn.model_selection import KFold
import datetime
import pandas as pd
import numpy as np
import random
import math as math
import itertools
import sys

eval_feature_sel = []

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

## List of categorical features
listCategFeat = []

## List of continuous features
listContFeat = []


##***********************************EVALUATION*******************************************************

def kNearestNeigh(x_train, y_train, x_test):
    modelName = "KNN"
    x_train = np.nan_to_num(x_train)
    y_train = np.nan_to_num(y_train)
    for data in train_df:
        currentDBClusterSubspace.append(data)

    classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    classifier.fit(x_train, y_train.ravel())
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train.ravel(), cv=10)
    print('Accuracy:', accuracies.mean())
    y_pred = classifier.predict(x_test)
    evaluateModel(y_test, y_pred, modelName)


# def decisionTree(x_train, y_train, x_test):
#     modelName = "Decision Tree"
#     classifier = DecisionTreeClassifier()
#     classifier.fit(x_train, y_train)
#     y_pred = classifier.predict(x_test)
#     evaluateModel(y_test, y_pred, modelName)


## Evaluate the model
def evaluateModel(y_test, y_pred, modelName):
    ## Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    total = sum(sum(cm))

    print("Model Name is : ", modelName)

    ## Calculate accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    print('Accuracy:', accuracy)

    ## Calculate sensitivity
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity:', sensitivity)

    ## Calculate specificity
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('Specificity:', specificity)

    ## Calculate F-measure
    ## Average can be 'micro','weighted' or 'None'
    f_measure = f1_score(y_test, y_pred, average='macro')
    print('F Measure:', f_measure)

    # Logging
    with open('DressEvaluation.txt', 'a') as f:
        print("", file=f)
        print("Model Name is : ", modelName, file=f)
        print('Accuracy:', accuracy, file=f)
        print('Sensitivity:', sensitivity, file=f)
        print('Specificity:', specificity, file=f)
        print('F Measure:', f_measure, file=f)
        print("******************-- End of Iteration of K fold --*******************", file=f)

    # AUC Score


#    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
#    metrics.auc(fpr, tpr)

# ******************************************************END EVALUATION**********************************************


## Generate constraint pairs
def createRandomConsPairs(listCons, n):
    ## Generate all possible non-repeating pairs
    pairsCons = list(itertools.combinations(listCons, 2))

    ## Randomly shuffle these pairs
    random.shuffle(pairsCons)

    ## Randomly pick and return required no of pairs
    return random.sample(pairsCons, n)


## Create a list of all ML constraints
def createMLConsList(dataFrame):
    global posMLCons
    global negMLCons

    posMLCons.clear()
    negMLCons.clear()
    #print("DataFrame", dataFrame)
    for index, row in dataFrame.iterrows():
        #print("Index", index)
        if float(row["mrt_liverfat_s2"]) == 0:
            negMLCons.append(index)

        if float(row["mrt_liverfat_s2"]) == 1:
            posMLCons.append(index)
    #print("negMLCons", negMLCons)
    #print("posMLCons", posMLCons)
    return createRandomConsPairs(posMLCons, int(noMLCons / 2)) + createRandomConsPairs(negMLCons, int(noMLCons / 2))


## Create a list of (pairs of) all NL constraints
def createNLConsList():
    global posNegNLCons

    posNegNLCons.clear()

    #print("posMLCons", posMLCons)
    #print("negMLCons", negMLCons)
    for i, j in zip(posMLCons, negMLCons):
        tupNLCons = (i, j)
        posNegNLCons.append(tupNLCons)
    #print("posNegNLCons", posNegNLCons)

    ## Create a list of (pairs of) NL constraints
    return random.sample(posNegNLCons, noNLCons)


## Check whether the feature is categorical or continuous
def checkFeatType(feature, listTypeFeat):
    if feature in listTypeFeat:
        #        print(feature, 'is in', listTypeFeat)
        return True
    else:
        return False


## Calculate the distance between object pairs in a feature
def calculateSqDistDiff(feature, objPairX, objPairY):
    diffDist = 0
    #    print("Inside calculateSqDistDiff")
    #    print(feature, objPairX, objPairY)
    ## If both oject pairs are categorical
    if checkFeatType(feature, listCategFeat):
        if math.isnan(objPairX):
            if math.isnan(objPairY):
                diffDist = 1
            else:
                diffDist = 1
        else:
            if math.isnan(objPairY):
                diffDist = 1
            else:
                if objPairX == objPairY:
                    diffDist = 0
                else:
                    diffDist = 1
    #        print(diffDist)

    ## If both object pairs are continuous
    if checkFeatType(feature, listContFeat):
        if math.isnan(objPairX):
            if math.isnan(objPairY):
                diffDist = 1
            else:
                diffDist = 1
        else:
            if math.isnan(objPairY):
                diffDist = 1
            else:
                diffDist = objPairX - objPairY
        # print(diffDist)

    # print("Feature:",  feature)
    # print("objPairX = ", objPairX)
    # print("objPairY = ", objPairY)

    return (diffDist ** 2)


## Calculate distance between object pairs for every feature in a feature space using Heterogeneous Euclidean Overlap Metric
def calculateHEOM(subspace, objPairX, objPairY):
    sumDistSq = 0
    ## Calculate distance between object pairs for every feature in a feature space using Heterogeneous Euclidean Overlap Metric
    for feature in subspace.columns:
        # print("Subspace is: " , subspace)
        #print("objPairX", objPairX)
        #print("objPairY", objPairY)
        #print('feature', feature)

        #print('subspace.loc[objPairX][feature]', subspace.loc[objPairX][feature])
        #print('subspace.loc[objPairY][feature]', subspace.loc[objPairY][feature])
        sumDistSq = sumDistSq + calculateSqDistDiff(feature, subspace.loc[objPairX][feature],
                                                    subspace.loc[objPairY][feature])

    return math.sqrt(sumDistSq)


## Calculate the average distance between object pairs in a subspace
def calculateAvgDist(subspace, listConsPairs):
    totalDist = 0

    for consPair in listConsPairs:
        totalDist = totalDist + calculateHEOM(subspace, consPair[0], consPair[1])

    ## Total no of ML/NL constraints
    noConst = len(listConsPairs)

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
    listCommCluster = []

    for constPair in listNLConsPairs:
        listCommCluster.clear()
        listClusterCons.clear()

        for clusterNo in range(len(position_list)):
            # print('Cluster No: ', clusterNo)
            # print('elements in cluster: ', position_list[clusterNo])
            if constPair[0] in position_list[clusterNo]:
                listClusterCons.append(clusterNo)
            # print('listClusterMLCons: ', listClusterCons)
            if constPair[1] in position_list[clusterNo]:
                listClusterCons.append(clusterNo)
            # print('listClusterNLCons: ', listClusterCons)

        # listCommCluster = list(set(listClusterMLCons).symmetric_difference(set(listClusterNLCons)))
        listCommCluster = list(set(listClusterCons))
        # print('listCommCluster: ',listCommCluster)
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

    ## No of satisfied NL constraints
    noSatisNL = calculateNoSatisNLCons()
    # print('No of satis NL: ', noSatisNL)

    ## Total no of ML constraints
    totalNoML = len(listMLConsPairs)

    ## Total no of NL constraints
    totalNoNL = len(listNLConsPairs)

    ## Quality score based on constraint satisfaction
    qualScoreConst = (noSatisML + noSatisNL) / (totalNoML + totalNoNL)

    return qualScoreConst


## Calculate the quality score of each subspace based on the clustering performed by DBSCAN algorithm.
# Input:
# Output: Quality score for each subspace.
# def calculateSubspaceScore(subspace):
## Quality score based on constraint satisfaction
#    constraintScore = calculateConstScore()

## Quality score based on distance
#    distanceScore = calculateDistScore(subspace)

#    if distanceScore < 0:
#        negDistSubspace.append(subspace.head(0))

## Final quality score
#    finalScore = constraintScore * distanceScore

#    return finalScore


## Set file path
## Test Data
# TRAIN_PATH = '/media/sumit/Entertainment/OVGU - DKE/Summer 2018/DRESS/csv_result-ship_14072018.csv'

## Original (Labeled + Unlabeled) Data
# TRAIN_PATH = '/media/sumit/Entertainment/OVGU - DKE/Summer 2018/DRESS/csv_result-ship_22042018.csv'

## Labeled Data
TRAIN_PATH = '/home/kundu/Desktop/csv_result-ship_labeled_data.csv'


## Load data from file
def loadDatasetWithPandas(path):
    rawData = pd.read_csv(path)

    # replacing the string indicating missing values with the numpy value for missing values
    # NaNProcessedData = rawData.replace({'na': np.nan}, regex=True)
    NaNProcessedData = rawData
    return NaNProcessedData


## Create new subspace from the previously selected best subspace
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


negDistSubspace = []


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
                    featureSpace.append(k)
            else:
                featureSpace.append(i)
    else:
        featureSpace.append(subspace)

    candidateDataFrame = pd.DataFrame(trainData, columns=featureSpace)

    print(" ")
    print("Subspace:", candidateDataFrame.columns)

    ## Log the subspace along with execution Date and Time Stamp
    with open('output.txt', 'a') as f:
        print("", file=f)
        print("DateTime:", datetime.datetime.now(), file=f)
        print("Subspace:", candidateDataFrame.columns, file=f)

    ## Distabce Score of a subspace
    distScoreSubspace = calculateDistScore(candidateDataFrame)
    print("Distance Score:", distScoreSubspace)

    ## Mark subspaces having negative Distance Score
    if distScoreSubspace < 0:
        negDistSubspace.append(candidateDataFrame.columns)
        totalQualScoreSubspace = -10

        ## Log the total score of a subspace
        with open('output.txt', 'a') as f:
            print("Total Score: Negative Distance Score", file=f)

        return totalQualScoreSubspace

    ## Constraint Score of a subspace
    constScoreSubspace = createDBCluster(candidateDataFrame)

    ## Total Score of a subspace
    totalQualScoreSubspace = distScoreSubspace * constScoreSubspace
    print("Total Score:", totalQualScoreSubspace)

    ## Log the total score of a subspace
    with open('output.txt', 'a') as f:
        print("Total Score:", totalQualScoreSubspace, file=f)

    return totalQualScoreSubspace


## Custom distance function
def myDistance(x, y):
    dist = 0

    for i in range(len(currentDBClusterSubspace)):
        dist = dist + calculateSqDistDiff(currentDBClusterSubspace[i], x[i], y[i])

    return math.sqrt(dist)


currentDBClusterSubspace = []

position_list = []




## Create clusters using DBSCAN Algorithm
def createDBCluster(candidateDataFrame):
    position_list.clear()

    currentDBClusterSubspace.clear()

    for data in candidateDataFrame:
        currentDBClusterSubspace.append(data)

    epsilon = calcEpsilon(candidateDataFrame)
    print("Epsilon:", epsilon)

    ## Fit the data
    db = DBSCAN(eps=epsilon, min_samples=minPts, metric=myDistance).fit(candidateDataFrame)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    lable_list = []

    a = 0;
    for i in labels:
        lable_list.append(i)

    output = []
    for x in lable_list:
        if x not in output:
            output.append(x)

    print("Unique Values:", output)

    for k in output:
        if k != -1:
            a = 0
            custer_list = []
            for l in lable_list:

                if l == k:
                    custer_list.append(a)
                a = a + 1

            position_list.append(custer_list)

    ## Number of clusters formed (ignoring noise points)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("No of Clusters:", n_clusters_)

    ## Constraint Score of a subspace
    constScore = calculateConstScore()
    print("Constraint Score:", constScore)

    return constScore


## Append the score against the respective feature. Example: [rbc, 0.5]
def scoreCalculate(ssList):
    print("Subspace List:", ssList)

    subspaceScoresList = []

    for feature in ssList:
        finalSubspaceScore = performDBScan(feature)

        subspaceScoresList.append([feature, finalSubspaceScore])

    return subspaceScoresList


def bestScore(ssScoresList):
    return max_by_score(ssScoresList)


## Filter out features/subspaces having negative Distance Score
def dropNegativeScoreFeature(ssScoreList):
    negFeatureList = []

    print("Score List: ", ssScoreList)
    print("Size of Score List: ", len(ssScoreList))

    for i in ssScoreList:
        if i[1] < 0:
            negFeatureList.append(i)
            print("Negative Score:", i)

    return negFeatureList


## Search the Traverses the list to get the best score(Max score)
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

    while previousBestScore < currentBestScore:
        possible_sslist = makeSubspaces(df, feat_select)
        score_sslist = scoreCalculate(possible_sslist)
        featureset_score = bestScore(score_sslist)

        ##Filter Addition
        negFeatures = dropNegativeScoreFeature(score_sslist)

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
        global eval_feature_sel
        eval_feature_sel = features_selected_final
        # print(possible_sslist)
        # print('')
        # print(score_sslist)
        # print('')
        print("Subspace with best score for the current iteration:")
        print(featureset_score)

        print("Features to be considered for next iteration:")
        print(features_for_nxt_iter)
        print("################################")
        print('Features selected Final:')
        print(features_selected_final)
        print("################################")
        print("Number of features left for next iteration:", len(features_for_nxt_iter))
        print("Previous Best Subspace Score:", previousBestScore)
        print("Current Best Subspace Score:", currentBestScore)

        # Logging
        with open('output.txt', 'a') as f:
            print("", file=f)
            print("Datetime", datetime.datetime.now(), file=f)
            print("Features selected:", features_selected_final, file=f)
            print("-- End of Iteration --", file=f)

        iter_subspace(df[features_for_nxt_iter], features_selected_final, previousBestScore, currentBestScore)

        #print("Inside loop: ", eval_feature_sel)
        break


    #print("Outside loop: ", eval_feature_sel)
    #return eval_feature_sel






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


## Calculate the value of Min Points
def calcMinPts():
    count = trainData.shape[0]

    D = math.log(count)
    minPts = int(D)
    return minPts


## Calculate the value of Epsilon
def calcEpsilon(currentSubspace):
    data = currentSubspace

    minPts = calcMinPts()
    kneighbour = minPts - 1
    nbrs = NearestNeighbors(n_neighbors=minPts, algorithm='auto', metric=myDistance).fit(data)
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


##
def NormalizeData(inputdataframe, columnName):
    scaler = pp.MinMaxScaler(feature_range=(0, 1))
    null_index = inputdataframe[columnName].isnull()
    inputdataframe.loc[~null_index, [columnName]] = scaler.fit_transform(inputdataframe.loc[~null_index, [columnName]])
    return inputdataframe


def dataPreprocessing(dataFrame):
    ## Set class variable to 0 for 'Neg' and 1 for 'Pos'
    for index, row in dataRaw.iterrows():
        if float(row["mrt_liverfat_s2"]) <= 10:
            dataRaw.mrt_liverfat_s2.iloc[[index]] = 0

        if float(row["mrt_liverfat_s2"]) > 10:
            dataRaw.mrt_liverfat_s2.iloc[[index]] = 1

    ## Delete the unwanted features such as ones have date, time and id stored in it
    dataFrame = dataFrame[dataFrame.columns.difference(
        ['id', 'exdate_ship_s0', 'exdate_ship_s1', 'exdate_ship_s2', 'exdate_ship0_s0', 'blt_beg_s0', 'blt_beg_s1',
         'blt_beg_s2'])]

    ## Replace '?' with 'NaN'
    dataFrame = dataFrame.replace('?', np.NaN)

    k = dataFrame.nunique()
    j = pd.unique(dataFrame.columns.values)

    uniqueValFtreList = []

    a = 0;
    b = 0;
    for i in k:
        b = 0
        for l in j:
            if a == b:
                uniqueValFtreList.append([l, i])
            b = b + 1
        a = a + 1

    # print("***************Feature with possible Categorial Data*******************")
    for x in uniqueValFtreList:
        if x[1] <= 10:
            listCategFeat.append(x[0])
        else:
            listContFeat.append(x[0])

    ## List of text based categorical features
    listTextCategFeat = ['mort_icd10_s0', 'stea_alt75_s0', 'stea_alt75_s2', 'stea_s0', 'stea_s2']

    ## List of text based categorical features having missing values (np.NaN)
    listTextCategFeatNaN = []

    ## List containing index of NaN for text based categorical features
    listTextCategFeatIndexNaN = []

    ##
    for data in dataFrame.columns:
        if data not in listTextCategFeat and dataFrame[data].dtype == 'O':
            dataFrame[[data]] = dataFrame[[data]].apply(pd.to_numeric)

        if dataFrame[data].dtype == 'O' and data in listTextCategFeat:
            unique_elements = dataFrame[data].unique().tolist()

            if np.nan in unique_elements:
                listTextCategFeatNaN.append(data)
                listTextCategFeatIndexNaN.append(len(unique_elements) - 1)
                unique_elements.remove(np.nan)
                unique_elements.append(np.nan)

            dataFrame[data] = dataFrame[data].apply(lambda x: unique_elements.index(x))

    for col in dataFrame.columns:
        if col in listTextCategFeatNaN:
            dataFrame[col] = dataFrame[col].replace(listTextCategFeatIndexNaN[listTextCategFeatNaN.index(col)], np.NaN)

    ## Normalize continuous variables
    for col in dataFrame.columns:
        if col in listContFeat:
            dataFrame = NormalizeData(dataFrame, col)

    return dataFrame


# **********************************Main Function*************************************************

## Load the entire dataset into a data frame
dataRaw = loadDatasetWithPandas(TRAIN_PATH)

preTrainData = dataPreprocessing(dataRaw)

preTestData = preTrainData
trainData = preTrainData

print(preTrainData)

# prepare cross validation
kfold = KFold(5, True, 1)

# trainlist
# testlist

countItter = 0
# enumerate splits
# Looping for each fold of data
for train, test in kfold.split(preTrainData):

    #print("PreTestData", preTestData)
    #print("trainData", trainData)
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")





    testData = trainData.drop(train)
    trainData = preTestData.drop(test)
    trainDataEvaluation = trainData

    #print("testData", testData)
    #print("trainData", trainData)

    # **************************************************Calling DRESS****************************************
    ## User sets the no of ML constraints
    # noMLCons = input("Enter the number of must-link constraints to be used:")
    noMLCons = 10
    ## User sets the no of NL constraints
    # noNLCons = input("Enter the number of not-link constraints to be used:")
    noNLCons = 10
    #print("train index", train)
    #print("test index", test)
    ## List of randomly selected ML constraint pairs
    listMLConsPairs = createMLConsList(trainData)
    # listMLConsPairs = [(126, 160), (21, 503), (238, 433), (127, 521), (422, 512), (35, 212), (212, 267), (391, 396), (71, 252), (4, 465)]
    # listMLConsPairs = [(69, 422), (469, 561), (144, 261), (505, 569), (109, 176), (304, 385), (111, 480), (196, 387), (331, 491), (101, 447)]
    print("listMLConsPairs: ", listMLConsPairs)
    ## List of randomly selected NL constraint pairs
    listNLConsPairs = createNLConsList()
    # listNLConsPairs = [(393, 117), (28, 6), (219, 88), (21, 2), (41, 12), (13, 1), (239, 95), (207, 85), (155, 68), (134, 53)]
    # listNLConsPairs = [(406, 128), (232, 93), (223, 91), (413, 129), (206, 84), (218, 87), (563, 200), (150, 63), (545, 196), (47, 16)]
    print("listNLConsPairs: ", listNLConsPairs)
    ## Delete the unwanted features such as ones have date, time, id and class label stored in it
    trainData = trainData[trainData.columns.difference(['mrt_liverfat_s2'])]

    # epsilon = calcEpsilon()

    minPts = calcMinPts()

    # The main function of the program.
    currentBestScore = 0.0000001
    previousBestScore = 0
    features_selected = []

    print("The scores are: ", currentBestScore, "    ", previousBestScore)
    # List of features selected by Dress
    iter_subspace(trainData, features_selected, previousBestScore, currentBestScore)
    featureSelectedDress = eval_feature_sel
    #featureSelectedDress = final_subspace
    #print(final_subspace)
    print("*****************************************************************************************")
    print(eval_feature_sel)

    # Logging
    with open('DressEvaluation.txt', 'a') as f:
        print("", file=f)
        print("Datetime", datetime.datetime.now(), file=f)
        print("Final Features selected:", featureSelectedDress, file=f)
        print("Itteration Number:", countItter, file=f)

    # ************************************Preparing That fold of Data for Evaluation*********************************
    ## Create input data frame for evaluation based on feature set obtained from DRESS
    train_df = pd.DataFrame(data=trainDataEvaluation, columns=featureSelectedDress)
    test_df = pd.DataFrame(data=testData, columns=featureSelectedDress)

    # Logging
    with open('DressEvaluation.txt', 'a') as f:
        print("", file=f)
        print("Datetime", datetime.datetime.now(), file=f)
        print("Shape of Train Data:", train_df.shape, file=f)
        print("Shape of Test Data:", test_df.shape, file=f)

    ## Create training set for target variable

    # Training data with Selected Features of DRESS
    x_train = train_df
    x_train = x_train.values

    # Target Variable of Training data.
    y_train = pd.DataFrame(data=trainDataEvaluation, columns=['mrt_liverfat_s2'])
    y_train = y_train.values

    # Test data with Selected Features of DRESS
    x_test = test_df
    x_test = x_test.values

    # Target Variable of Test data.
    y_test = pd.DataFrame(data=testData, columns=['mrt_liverfat_s2'])
    y_test = y_test.values

    kNearestNeigh(x_train, y_train, x_test)

    # *************************************Resetting the Dataframes for next itteration of K fold*******************

    preTestData = preTrainData
    trainData = preTrainData
    countItter = countItter + 1

    # ****************************************************Finish***********************************************




