import pandas as pd;
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

# Pre-processing
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

# base['name'].value_counts()
base = base.drop('name', axis = 1) # there is a bunch of different name
# base['seller'].value_counts()
base = base.drop('seller', axis = 1) # almost all of those register is private
# base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)  # same case above

base['vehicleType'].value_counts()

# # Removing lower nonsense prices
i1 = base.loc[base.price <= 10]
base = base[base.price > 10]

# Removing higher nonsense prices
i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

# Looking for null values
# We're gonna use the approch to repeat the most often on null values
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine

base.loc[pd.isnull(base['gearbox'])] 
base['gearbox'].value_counts() # manuell

base.loc[pd.isnull(base['model'])] 
base['model'].value_counts() # golf
 
base.loc[pd.isnull(base['fuelType'])] 
base['fuelType'].value_counts() # benzin

base.loc[pd.isnull(base['notRepairedDamage'])] 
base['notRepairedDamage'].value_counts() # nein

values = {
        'vehicleType': 'limousine',
        'gearbox': 'manuell',
        'model': 'golf',
        'fuelType': 'benzin',
        'notRepairedDamage': 'nein'
    }

base = base.fillna(value = values )

predictors = base.iloc[:,1:13].values
real_price = base.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_predictors = LabelEncoder() 

predictors[:,0] = labelencoder_predictors.fit_transform(predictors[:,0])
predictors[:,1] = labelencoder_predictors.fit_transform(predictors[:,1])
predictors[:,3] = labelencoder_predictors.fit_transform(predictors[:,3])
predictors[:,5] = labelencoder_predictors.fit_transform(predictors[:,5])
predictors[:,8] = labelencoder_predictors.fit_transform(predictors[:,8])
predictors[:,9] = labelencoder_predictors.fit_transform(predictors[:,9])
predictors[:,10] = labelencoder_predictors.fit_transform(predictors[:,10])


onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
predictors = onehotencorder.fit_transform(predictors).toarray()

predictors[0]