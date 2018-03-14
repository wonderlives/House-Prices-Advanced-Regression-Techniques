### Create clusters and perform cluster based regression
### Author: Roger Ren
### Date: 03/06/18

# Import libs
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import collections

def getPCAcomponent(df, type = "Train", threshold = 0.9):
    """
    Returns returns the PCA fit matrix of the input dataframe, 
    using minimum PCAs that can meet the threshold requirement
    """

    # Check if df is a valid dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The input is not a valid dataFrame.')

    # Check if the threshold is within 1

    # Make adjustment if necessary, also check input type is valid
    if type == "Train":
        # Remove SalePrice before PCA.
        try:
            df.drop(['SalePrice'], axis = 1, inplace = True)
        except:
            print("There was no 'SalePrice' to drop in the 'train', continue....")
    elif type == "Test":
        pass
    else:
        raise ValueError("Type has to be 'Train' or 'Test'.")

    # Scale the data
    dfToScaleMatrix = preprocessing.scale(df.as_matrix())

    # Perform PCA
    pca = PCA().fit(dfToScaleMatrix)

    # Get incremental and cumulative viariance explained by PCA.
    increVarExpl = pca.explained_variance_ratio_
    totlSumVar = np.array([sum(increVarExpl[0:i+1]) for i,x in enumerate(increVarExpl)]) 

    # Get the num of PCA needed.
    for i, val in enumerate(totlSumVar):
        if val >= threshold:
            numPCA = i
            break

    # Fit transform the df
    result = PCA(n_components=numPCA).fit_transform(dfToScaleMatrix) 
    return result


def getBestClusterNum(pcaFitMatrix, maxCluster=30, method = "KMeans"):
    """
    Returns the best number of clusters to use based on different distancce based scores
    """

    # Num of clusters to consider
    clustersNumList = range(2,maxCluster)

    # Score distance metric
    distanceList = [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

    # Cluster method
    clusterMethods = ['K-Means', 'Affinity propagation', 'Mean-shift', 'Spectral clustering', 
    'Ward hierarchical clustering', 'Agglomerative clustering', 'DBSCAN', 'Gaussian mixtures', 'Birch']

    
    fit_results_list = []
    for i in clusters:
        fit_result = KMeans(n_clusters=i, init='k-means++', n_init=50, max_iter=2000).fit(full_dataset_after_PCA)
        fit_results_list.append(fit_result)



def putClusterIndex(df, nCluster = 2, method = "KMeans", splitFile = False, writeToFile = False, outPut = 'Train'):
    
    # PCA stage
    print("Step 1: Perform PCA..")
        # Copy salePrice and save it somewhere else
    if outPut == "Train":
        salePriceCol = df['SalePrice'].copy()
        df.drop(['SalePrice'], axis = 1, inplace = True)
    elif outPut == "Test":
        pass
    else:
        return ("You have to choose either 'Test' or 'Train' as output.")
        #  Scale the rest cols
    trainMatrix = df.as_matrix()
    trainMatrix = preprocessing.scale(trainMatrix)
    dfPCA = PCA(n_components=60).fit_transform(trainMatrix) 
    
    # Clustering stage
    print("Use {0} to perform clustering..".format(method))
        # Perform clustering 
    if method == 'KMeans':
        clusterResult = KMeans(n_clusters=nCluster, init='k-means++', n_init=50, max_iter=2000).fit(dfPCA)
    if method == 'Agglo':
        clusterResult = AgglomerativeClustering(n_clusters=nCluster).fit(dfPCA) 
        # Add cluster indexes to the dataframe
    df['Clusterlabel'] = clusterResult.labels_
        # Print out the clusters and items count in each cluster
    clusterStats = collections.Counter(clusterResult.labels_)
    
    # Print out results
    print("Clustering Results:")
    for key, value in clusterStats.items():
        print("There is {0} items in Cluster #{1}".format(key, value))
    
    # Data output
        # Add SalePrice back to the dataframe
        if outPut == "Train":
            df['SalePrice'] = salePriceCol
        # When NOT split files according on cluster index:
    if splitFile == False and writeToFile == True:
        print("Save ONE file to local...")
        print("Save to ./data/FinalRR{0}_{1}_Clustered{2}_Dense_OF_Log.csv".format(method, outPut, nCluster))
        df.to_csv("./data/FinalRR{0}_{1}_Clustered{2}_Dense_OF_Log.csv".format(method, outPut, nCluster),index=False)
        # When DO SPLIT files according to cluster index:
    if splitFile == True and writeToFile == True:
        print("Save MULTIPLE files to local...")
        for cluster in np.unique(clusterResult.labels_):
            temp = df[df['Clusterlabel']==cluster].copy()
            temp.drop(['Clusterlabel'], axis = 1, inplace = True)
            print("Save to ./data/FinalRR{0}_Cluster{1}_{2}_Dense_OF_Log.csv".format(method, cluster, outPut))
            temp.to_csv("./data/FinalRR{0}_Cluster{1}_{2}_Dense_OF_.Log.csv".format(method, cluster, outPut),index=False)
    print("Program Finished!")

if __name__ == "__main__":
    # execute only if run as a script
    train = pd.read_csv('./data/train_120feats_Dense_OutlierFree_LogTransform.csv')
    test = pd.read_csv('./data/test_119feats_Dense_OutlierFree_LogTransform.csv')
    test()
    # optimize()
    