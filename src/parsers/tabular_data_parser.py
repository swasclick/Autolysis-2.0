import pandas as pd
import numpy as np
import json
from llm.api_caller import APICaller
from llm.prompts.prompt_loader import PromptLoader
from modules.logging_module import Logger
from modules.error_module import ResponseParseError
logger = Logger().get_logger()

'''
flow:
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

class LLMTabularParser:
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
    def __init__(self, creds: dict, backend: str ='ollama')  -> None:
        self.column_classes = {}
        self.column_metadata = {}
        self.backend = backend
        self.creds = creds

    def get_column_groups(self, df: pd.DataFrame) -> str:
        
        df_context = df.head(5).to_string().strip()
        prompt = PromptLoader().load_prompt('classify_columns') + df_context
        caller = APICaller(provider = self.backend)
        classification = caller.make_api_call(prompt)
        try:
            json.loads(classification['response'])
            logger.debug('Got API response')
        except Exception as e:
            logger.error(f"Couldn't parse LLM classification JSON: error {e}")
            raise ResponseParseError()
        return classification
    
    def validate_column_classification(self, classification: json, df: pd.DataFrame) -> tuple:
        
        df_context = df.tail(5).to_string().strip()
        prompt = PromptLoader().load_prompt('validate_classification') + df_context
        caller = APICaller(provider = self.backend)
        validated_classification = caller.make_api_call(prompt)
        try:
            json.loads(validated_classification['response'])
            logger.debug('Got API response')
        except Exception as e:
            logger.error(f"Couldn't parse LLM classification JSON: error {e}")
            raise ResponseParseError()

        return (True, validated_classification) if validated_classification['errors'] == [] else (False, validated_classification)

    
    def fit(self, df: pd.DataFrame):
        pass

    
    
    def get_metadata(self, series: pd.Series, col_name: str):
        pass

    def check_missing(self, series: pd.Series, col_name: str):
        metadata = {}
        
        metadata['missing_count'] = series.isna().sum()