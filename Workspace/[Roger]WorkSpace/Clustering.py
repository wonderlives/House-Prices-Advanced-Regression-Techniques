from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.decomposition import PCA
def putClusterIndex(df, nCluster = 2, method = "KMeans"):
    # PCA stage
        # Copy salePrice and save it somewhere else
    salePriceCol = df['SalePrice'].copy()
    df.drop(['SalePrice'], axis = 1, inplace = True)
        #  Scale the rest cols
    trainMatrix = df.as_matrix()
    trainMatrix = preprocessing.scale(trainMatrix)
    dfPCA = PCA(n_components=60).fit_transform(train) 
    
    # Clustering stage
        # Perform clustering 
    if method == 'KMeans':
        clusterResult = KMeans(n_clusters=2, init='k-means++', n_init=50, max_iter=2000).fit(dfPCA)
    if method == 'Agglo':
        clusterResult = AgglomerativeClustering(n_clusters=2).fit(dfPCA) 
        # Add cluster indexes to the dataframe
    df['Clusterlabel'] = clusterResult.labels_
    
    # Data output prepare
        # Create add price back to the df
    df['SalePrice'] = salePriceCol
        # Split the clusters
    trainCluster0 = df[df['Clusterlabel']==0].copy()
    trainCluster1 = df[df['Clusterlabel']==1].copy()
        # Drop Cluster labels
    trainCluster0.drop(['Clusterlabel'], axis = 1, inplace = True)
    trainCluster1.drop(['Clusterlabel'], axis = 1, inplace = True)
        # Get count of points
    print("{0} points in Cluster #1 and {1} points in Cluster #2".format(trainCluster0.shape[0], trainCluster1.shape[0]))
        # Save the files to local
    trainCluster0.to_csv(method+"_0_train_120feats_Dense_OutlierFree_NoTransform.csv")
    trainCluster1.to_csv(method+"_1_train_120feats_Dense_OutlierFree_NoTransform.csv")

putClusterIndex(trainLog, nCluster = 2, method = "KMeans")
putClusterIndex(trainLog, nCluster = 2, method = "Agglo")