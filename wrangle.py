#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import env
import pandas as pd
import numpy as np
import seaborn as sns


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

    # Drop all rows with NaN values.
    df = df[df['calculatedfinishedsquarefeet'].notna()]

    # Convert all columns to int64 data types.
    df = df.astype('float')
    
    # Drop unwanted columns
    df = df.drop(columns='Unnamed: 0')

    return df

