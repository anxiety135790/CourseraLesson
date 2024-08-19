#%%

#%%
import  pandas as pd
import  numpy as np
import pip

#%% md
# 
#%%
source_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')

#return the dimensionality of the DataFrame
source_data.shape

#%%
source_data.describe()

#%%
dataisNull= source_data.isnull().sum()
#%%
#return Index of columns
dataCols = source_data.columns
print(dataCols)

#%%
print(source_data[dataCols])

#%%
predictors = source_data[dataCols[dataCols != 'Strength']]
predictors.head()

#%%
target = source_data['Strength']
target.head()
#%%
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
#%%
n_cols = predictors_norm.shape[1]
print(n_cols)
#%%

#%%
try:
    import tensorflow
    import keras
except ImportError:
    import  pip
    pip.main(["install", "tensorflow"])
    pip.main(["install", "keras"])
    from keras.models import  Sequential    
    from keras.layser import  Dense
    


#%%
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
#%%
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
#%%
