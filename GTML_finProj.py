#################################################################################################################################################
## Kevin Mueller, Georgia Tech Final project, 12-11-2014.                                                                                      ##
##                                                                                                                                             ##
## Depedincies: Scikit-learn, Numpy, scipy, possibly others. A core Anaconda installation should have all necessary                            ##
## modules.                                                                                                                                    ##                                                                                                                                
##                                                                                                                                             ##
## Read Me: In order to run code first scroll down to the BuildandRunKNN function and change the path variable to                              ##
## path where data is located. Next, Scroll down to main scipt. There change C and nNe to desired cluster size and                             ##
## Neighbor size respectivefully. It should run bug free. Depending on resources it should take around 15 mins.                                ##
##                                                                                                                                             ##
## Output: Default is a prediction csv file and sum of squred error metric. pred.csv is saved in default path. 
## Unfortunately I took the MSE (mean squared error) instead of the root mean squared error (RMSE), however analysis is the same.              ##                                                                                                             ##           
#################################################################################################################################################



# function for calculating pearson correlation
def pcs2(dic,x, y):
    from numpy import sqrt
    
   #find all mutually rated items 
   # shared_items = []
    #for i in range(1,n_c+1):
        #if np.isnan(Umat.iloc[x-1,i]) == False and np.isnan(Umat.iloc[y-1,i])  == False:
            #shared_items.append(i)
            
    # Get the list of mutually rated items
    shared_items={}
    for item in dic[x-1]:
        if item in dic[y-1]: shared_items[item]=1        
            
    # number of elements
    n = len(shared_items)
    
    if n == 0: return 0
    
    val1 = [dic[x-1][it] for it in shared_items]
    val2 = [dic[y-1][it] for it in shared_items]
    
    r = np.corrcoef(val1,val2)
    r = r[0,1]
    if np.isnan(r): r = 0
        
    

    #if den==0: return 0
    
    #r = num/den
    
    return r

#########################################################################################################
def guess(user_id, c_id, top_n,corr,Umat,means2):
    s = corr[user_id-1].argsort()

    # list of top n users
    #sim_users = s[::-1][:top_n] 
    sim_users = s[::-1][:top_n]
    
    score = 0
    ind = 0
    #u = 0
    
    for u in sim_users:
    #while ind < top_n:
        if np.isnan(Umat.iloc[u,c_id]): score += 0
        else:
            score += (Umat.iloc[u,c_id] - means2[u]) 
            ind += 1
        #u += 1
    
    # average the normalzied matrix for top n users


    if ind == 0:
        score = 0
    else:
        score /= ind
    

    # Add the average to unnormalize the matrix

    score += means2[user_id-1]

    return score

##################################################################################################################################################################################################################

def BuildandRun_knn(n_clusters,n_top):

    # set this parameter in order to load data from correct path
    path = '/Users/kevm1892/Downloads/'

    # function: handles loading the data, transforming it to panda data frames

    n_c = n_clusters



    # Load training data into a pandas data frame
    df = pd.read_csv(path + 'GT_Rec_Train.csv',header = None)
    df.columns = ['user id','movie id','rating','timestamp']

    # Load item info
    dfI = pd.read_csv(path + 'Item.csv',header = None)
    dfI.columns = ['movie id','movie title','release date','video release data','IMDB URL', 'unknown', 'action', 'adventure','animation','childrens','comedy', 'crime', 'documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war','western']

    # Load test data set

    dfT = pd.read_csv(path + 'test.csv',header = None)
    dfT.columns = ['user id','movie id','rating','timestamp']

    from IPython.display import display, clear_output

    # reorganize Movie dataset in order to build clusters based on genre
    dfI2 = dfI.iloc[:,range(5,(5+19))]



    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM

    #clf = KMeans(n_clusters = n_c)
    clf = GMM(n_components = n_c,covariance_type = 'tied') #comment out one or the other depending on if GMM or KMeans is used

    clf.fit(dfI2)

    out = clf.predict(dfI2)
    dfI['cluster'] = out + 1

    # get an index of an array for each cluster containg all possible movie ids
    arr = []
    arr2 = []
    for c in range(1,n_c+1):
        arr.append(dfI[dfI['cluster'] == c]['movie id'].values)

    clusters = np.zeros([len(df),1])

    # add to Data Frame

    for i in range(len(df)):


        for c in range(n_c):
            if df['movie id'][i] in arr[c]:
                clusters[i] = (c+1)
                break

    df['cluster'] = clusters  

    # the next step is to build a utility matrix as a pandas dataframe with the
    #following header:

        # index = user_id, cluster 1, cluster 2, ... , cluster N

    # the simplest way to accomplish this is to fill out the table in a brute force fashion

    Umat = pd.DataFrame()

    Umat['user_id'] = range(1,len(df['user id'].value_counts())+1)


    for c in range(1,n_c + 1):
        

        #C_rating = []
        C_rating = np.zeros([len(df['user id'].value_counts()),1])

        for i in range(1,len(df['user id'].value_counts())+1):
            is_user  = df['user id'] == i
            is_cluster = df['cluster'] == c
            C_rating[i-1] = (df[is_user & is_cluster]['rating'].mean())

        Umat['cluster' + str(c)] = C_rating   

    # mean for normalizing each rows ratings
    means2 = []
    for i in range(1,944):
        te = Umat.iloc[i-1,range(1,n_c+1)].mean()
        means2.append(te)

    # convert DF to dict for speed
    dic = Umat.iloc[:,range(1,n_c+1)].to_dict('records')

    # check for empty values (Nan) and delete them from dictionary
    for i in range(len(dic)):
        for key, item in dic[i].items():
            if np.isnan(item): 
                del dic[i][key]

    # build correlation matrix
    corr = np.zeros([len(dic),len(dic)])
    for i in range(1,len(dic)+1):
        for j in range(1,len(dic)+1):
            if i == j: corr[i-1,j-1] = -2
            else: 
                corr[i-1,j-1] = pcs2(dic,i,j)
                
    # find clusters for test set data            
    clusters = np.zeros([len(dfT),1])


    for i in range(len(dfT)):


        for c in range(n_c):
            if dfT['movie id'][i] in arr[c]:
                clusters[i] = (c+1)
                break
            
    dfT['cluster'] = clusters 

    # calculate the RMSE
    # for every row in test set

    se = 0
    pred = []
    for i in range(len(dfT)):
        pre = guess(int(dfT['user id'][i]),int(dfT['cluster'][i]),n_top,corr,Umat,means2)
        se += (round(pre,0) - dfT['rating'][i])**2
        pred.append(pre)

    # comment 2nd SE and uncomment first if using the RMSE metric. The report was done using default metric however. 
    #se = np.sqrt(se/len(dfT)) 
    se = se/len(dfT) 
    # code for building csv file from predictions
    dfP = pd.DataFrame()
    dfP['user id'] = dfT['user id']
    dfP['item id'] = dfT['movie id']
    dfP['rating'] = pred
    dfP['rating'] = dfP['rating'].round(4)
    dfP.to_csv(path + 'pred.csv',header = False,index = False)


    return se,pred


#########################################################################################################
#########################################################################################################

# Main script

# set C to desired cluster size and nNe to number of nearest neighbors, list format allows for multiple runs of
# program for comparison purposes.

c = [5]
nNe = [5]
predictions = []
ses = []
import numpy as np
import pandas as pd

# small script to calc rmse for different number of clusters
for ic in c:
    for nN in nNe:
        se,pr = BuildandRun_knn(ic,nN)
        ses.append(se)
        predictions.append(pr)
## Default is to print sum of squared error  
for se in ses:    
    print 'RMSE = {0:.3f}'  .format(se/100)