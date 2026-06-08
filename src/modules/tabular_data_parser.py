import pandas as pd
import numpy as np
from scipy.stats import entropy
from src.modules.api_caller import APICaller
from src.prompts.prompt_loader import PromptLoader

'''parse()
│
├── check_missing()
├── check_id()
├── check_boolean()
├── check_datetime()
│
├── check_email()
├── check_phone()
├── check_url()
│
├── check_geo()
│
├── check_numeric()
│   ├── classify_continuous()
│   ├── classify_discrete()
│   ├── classify_currency()
│   └── classify_percentage()
│
├── check_categorical()
│   ├── binary
│   ├── ordinal
│   ├── categorical_number
│   └── categorical_string
│
├── check_text()
│   ├── short
│   ├── long
│   ├── html
│   ├── json
│   ├── name
│   └── address
│
└── unknown'''


class TabularParser:
    '''
    Class to parse the columns into:
    
    0. ID

    # Numerical
    1. Numerical | Continuous
    2. Numerical | Discrete

    # Datetime
    3. Datetime | Yearly
    4. Datetime | Quarterly
    5. Datetime | Monthly
    6. Datetime | Weekly
    7. Datetime | Daily
    8. Datetime | Hourly
    9. Datetime | Minutely
    10. Datetime | Secondly

    # Categorical
    11. Categorical | String
    12. Categorical | Number
    13. Categorical | Binary
    14. Categorical | Ordinal

    # Text
    15. Text | Short
    16. Text | Long
    17. Text | HTML
    18. Text | JSON
    19. Text | Name
    20. Text | Address

    # Contact / Web
    21. Email
    22. Phone Number
    23. URL

    # Geographic
    24. Geographic | Country
    25. Geographic | State
    26. Geographic | City
    27. Geographic | Postal Code
    28. Geographic | Latitude
    29. Geographic | Longitude

    # Financial
    30. Currency
    31. Percentage

    # Logical
    32. Boolean

    # Files / Media
    33. File Path
    34. Image Path
    35. Audio Path
    36. Video Path

    # Missing / Unknown
    37. Missing / Empty
    38. Unknown

    # Targets
    39. Target | Regression
    40. Target | Binary Classification
    41. Target | Multi-Class Classification
    42. Target | Multi-Label Classification
    43. Target | Time Series Forecasting
    '''
    def __init__(self, creds: dict) -> None:
        self.column_classes = {}
        self.column_metadata = {}
        self.creds = creds

    def get_column_groups(self, df: pd.DataFrame, prompt: str):
        
        prompt = PromptLoader().load_prompt('classify_columns')
        caller = APICaller()
        response = caller.make_api_call(prompt)
    
    def fit(self, df: pd.DataFrame):
        pass

    def classify_column(self, series: pd.Series, col_name: str):
        pass
    
    def get_metadata(self, series: pd.Series, col_name: str):
        pass

    def check_missing(self, series: pd.Series, col_name: str):
        pass

    def check_id(self, series: pd.Series, col_name: str) -> dict:
        '''
        Checks whether a column is an identifier column
        '''
        
        series = series.dropna()
        n = len(series)
        
        if n == 0 : 
            return {
                'is_id': False,
                'groupable' : False,
            }
        
        # Check if ID in name
        norm_val_counts = series.value_counts(normalize=True)
        entropy_val = entropy(norm_val_counts, base=2)
        max_entropy = np.log2(len(norm_val_counts)) if len(norm_val_counts) > 1 else 1  
        
        condition1 = 'id' in col_name.lower() or '_id' in col_name.lower()
        condition2 = series.nunique() / len(series) > 0.9 # uniqueness ratio
        condition3 = entropy_val/max_entropy > 0.9 # measure of randomness 

        if condition1 and not condition2:
            pass
        
    def check_boolean(self, series: pd.Series, col_name: str):
        pass

    def check_datetime(self, series: pd.Series, col_name: str):
        pass

    def check_email(self, series: pd.Series, col_name: str):
        pass

    def check_phone(self, series: pd.Series, col_name: str):
        pass

    def check_url(self, series: pd.Series, col_name: str):
        pass

    def check_geo(self, series: pd.Series, col_name: str):
        pass

    def check_numeric(self, series: pd.Series, col_name: str):
        pass

    def check_categorical(self, series: pd.Series, col_name: str):
        pass

    def check_text(self, series: pd.Series, col_name: str):
        pass