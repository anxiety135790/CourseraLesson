#%% md
# ## K-Nearest Neighbors
#%% md
# 
#%%
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
except ImportError:
    import  pip
    pip.main(["install", "--upgrade", "pip"])
    pip.main(["install", "numpy","matplotlib","pandas","numpy","scikit-learn"])
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing 


#%% md
# 
#%%
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

#return top5 rows from dataframe in normal
df.head()

#%%
#check how many of each class in data
df['custcat'].value_counts()
#%%
#explore data using histogram
df.hist(column='income', bins=50)
#%%
df.columns
#%%
#df.columns get the index
X = df[df.columns.to_numpy()].values
X[0:5]
print(type(X))
#%%
y = df['custcat'].values
y[0:5]
#%%
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
#%% md
# 
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
#%% md
# 
#%%
from sklearn.neighbors import  KNeighborsClassifier

k = 4 
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
neigh
#%% md
# 
#%%
Yhat = neigh.predict(X_test)
Yhat[0:5]
#%%
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train,neigh.predict(X_train)))
print("Test set Accurancy: ", metrics.accuracy_score(Y_test, Yhat))
#%% md
# 
#%% md
# ### Practice
# 
# 
# 
# 
#%% md
# 
#%%

#%%


#%% md
# 
#%% md
# ## Decision Trees
#%% md
# 
#%%
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
#%%

#%%

#%% md
# 
#%% md
# 
#%% md
# ### Downloading Data
#%%
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
df.head()
#%%
df.shape
#%%

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]
#%%
from sklearn import  preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['low'.upper(), 'normal'.upper(), 'high'.upper()])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['normal'.upper(), 'high'.upper()])
#%%
X[:, 3] = le_Chol.transform(X[:,3])
X[0:5]
#%% md
# 
#%%
Y = df['Drug']
Y[0:5]
#%%

#%% md
# 
#%%
from sklearn.model_selection import  train_test_split
X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
drugTree = DecisionTreeClassifier(criterion='entropy' , max_depth=4)
drugTree.fit(X_trainset, Y_trainset)
#%%
predTree = drugTree.predict(X_testset)
#%% md
# 
#%%
print(predTree[0:5])
print(Y_testset[0:5])
#%% md
# ## Regression-Tree
#%%
# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
# install the missing python component 
try:
    import pandas as pd
    # Regression Tree Algorithm
    from sklearn.tree import DecisionTreeRegressor
    # Split our data into a training and testing data
    from sklearn.model_selection import train_test_split
except ImportWarning or ImportError:
    import pip
    pip.main(["install", "--upgrade", "pip"])
    pip.main(["install"])

    
#%%
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")
data.head()

#%%
# learn about the size of data
data.shape
#%%
#drop the rowsing with missing values 
#sum the na values
data.dropna(inplace=True)
data.isna().sum()
#%%
print(data.head(-1))
#%%
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
#%%
X.head()
#%%
Y.head()
#%% md
# #sklearn==1.5 mse ==> squared_error
# regression_Tree = DecisionTreeRegressor(criterion = "squared_error")
# regression_Tree.fit(X_train, Y_train)
# regression_Tree.score(X_test, Y_test)
#%%
