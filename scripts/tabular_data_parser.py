import pandas as pd
import numpy as np

class TabularParser:
    '''
    Class to label each column of the dataframe, which will decide how it is processed
    
    DATA TYPES
        1. Numerical | Continuous
        2. Numerical | Discrete
        3. Datetime | Yearly
        4. Datetime | Monthly
        5. Datetime | Daily
        6. Datetime | Hourly
        7. Datetime | Minutely
        8. Categorical | Number
        9. Categorical | String
        10. Texual
        '''
    
    
    def __init__(self):
        self.column_classes = {}
    
    def check_numerical(df: pd.DataFrame):
        '''
        Chack for numerical features
        '''
        
    def check_datetime(df:pd.DataFrame):
        '''
        Check for datetime features
        '''
        
    def classify_numerical(df: pd.DataFrame, numerical_cols: list[str]):
        '''
        Classify whether the numerical features are integers or continuous
        '''
        
    def check_categorical(df:pd.DataFrame):
        '''
        Check for categorical features
        '''
    
    def check_texual(df:pd.DataFrame):
        '''
        Check for texual features
        '''
        