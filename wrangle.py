#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import env
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def split_properties_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on fips.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on fips to get an even mix of properties in each county
    train_validate, test = train_test_split(df, test_size=.2, random_state=777)
    
    # splits train_validate into train and validate using train_test_split() stratifying on fips to get an even mix of properties in each county
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=777)
    return train, validate, test


def get_properties_data():
    filename = "properties_2017.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017', env.get_db_url('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

    
def impute_mean_taxvaluedollarcnt(train, validate, test):
    '''
    This function imputes the mean of the taxvaluedollarcnt column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['taxvaluedollarcnt'] = imputer.fit_transform(train[['taxvaluedollarcnt']])
    
    # transform age column in validate
    validate['taxvaluedollarcnt'] = imputer.transform(validate[['taxvaluedollarcnt']])
    
    # transform age column in test
    test['taxvaluedollarcnt'] = imputer.transform(test[['taxvaluedollarcnt']])
    
    return train, validate, test

def impute_mean_taxamount(train, validate, test):
    '''
    This function imputes the mean of the taxamount column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['taxamount'] = imputer.fit_transform(train[['taxamount']])
    
    # transform age column in validate
    validate['taxamount'] = imputer.transform(validate[['taxamount']])
    
    # transform age column in test
    test['taxamount'] = imputer.transform(test[['taxamount']])
    
    return train, validate, test

def wrangle_zillow():
    '''
    Read properties_2017 into a pandas DataFrame from mySQL,
    replace whitespaces with NaN values,
    drop any rows with Null values in calculatedfinishedsquarefeet,
    convert all columns to float,
    return cleaned properties DataFrame.
    '''

    # Acquire data

    properties = get_properties_data()

    # Replace white space values with NaN values.
    df = properties.replace(r'^\s*$', np.nan, regex=True)

    # only show rows where calculatedfinishedsquarefeet is not null.
    df = df[df['calculatedfinishedsquarefeet'].notna()]

    # Convert all columns to int64 data types.
    df = df.astype('float')
    
    # Drop unwanted columns
    df = df.drop(columns='Unnamed: 0')
    
    #split data
    train, validate, test = split_properties_data(df)
    
    #impute mean taxvaluedollarcnt where it is null
    train, validate, test = impute_mean_taxvaluedollarcnt(train, validate, test)
    
    #imput mean taxamount where it is null
    train, validate, test = impute_mean_taxamount(train, validate, test)

    return train, validate, test


# In[ ]:




