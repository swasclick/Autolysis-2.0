import pandas as pd
import numpy as np
import json
from llm.api_caller import APICaller
from llm.prompts.prompt_loader import PromptLoader
from modules.logging_module import Logger
from modules.error_module import ResponseParseError, ClassificationValidationError
logger = Logger().get_logger()

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
    def __init__(self, creds: dict = {}, backend: str = 'ollama', model: str = 'llama3')  -> None:
        self.column_classes = {}
        self.column_metadata = {}
        self.backend = backend
        self.creds = creds
        self.model = model

    def get_column_groups(self, df: pd.DataFrame) -> str:
        
        df_context = f"Columns: {df.columns} | Dataframe: {df.head(5).to_string().strip()}"
        prompt = PromptLoader().load_prompt('classify_columns') + df_context
        caller = APICaller(provider = self.backend)
        classification = caller.make_api_call(model=self.model, user_prompt=prompt)
        try:
            json.loads(classification['response'])
            logger.debug('Got API response')
        except Exception as e:
            logger.error(f"Couldn't parse LLM classification JSON: error {e}")
            raise ResponseParseError()
        return json.loads(classification['response'])
    
    def validate_column_classification(self, classification: dict, df: pd.DataFrame) -> tuple:
        
        # check if total number of columns is same
        attempt = 1
        while (list(classification.keys()) != list(df.columns)) and attempt < 6:
            logger.info("Total number of columns do not match, retrying...")
            classification = self.get_column_groups(df)
            attempt += 1
        if (list(classification.keys()) != list(df.columns)):
            logger.error("Cannot get the same number of columns after classification. Check model prompt?")
            raise ClassificationValidationError()
        
        # check classification via an LLM
        df_context = f"Columns: {df.columns} | Dataframe: {df.head(5).to_string().strip()}"
        prompt = PromptLoader().load_prompt('validate_classification') +"Dataset:"+ df_context + "Original Classification: " + json.dumps(classification) 
        caller = APICaller(provider = self.backend)
        validated_classification = caller.make_api_call(model=self.model, user_prompt=prompt)
        try:
            json.loads(validated_classification['response'])
            logger.debug('Got classification validation API response')
            
        except Exception as e:
            logger.error(f"Couldn't parse LLM classification JSON: error {e}")
            raise ResponseParseError()
        validated_classification = json.loads(validated_classification['response'])
        return (True, validated_classification) if validated_classification['errors'] == [] else (False, validated_classification)

    def get_metadata(self, df: pd.DataFrame, classification: dict):
        metadata = {}

        for column, column_type in classification.items():
            s = df[column]

            # Common metadata for every column
            column_meta = {
                "name": column,
                "dtype": str(s.dtype),
                "type": column_type,
                "row_count": len(s),
                "missing_count": int(s.isna().sum()),
                "missing_percentage": float(s.isna().mean() * 100),
                "unique_count": int(s.nunique(dropna=True)),
                "unique_percentage": float(
                    s.nunique(dropna=True) / len(s) * 100
                ) if len(s) > 0 else 0.0,
            }

            # -------------------------
            # ID
            # -------------------------
            if column_type == "ID":
                column_meta.update({
                    "is_unique": bool(s.is_unique),
                    "is_monotonic": bool(s.is_monotonic_increasing),
                    "sample_values": s.dropna().head(5).tolist(),
                })

            # -------------------------
            # Numerical
            # -------------------------
            elif column_type in [
                "Numerical | Continuous",
                "Numerical | Discrete"
            ]:
                numeric = pd.to_numeric(s, errors="coerce")

                column_meta.update({
                    "min": float(numeric.min()) if numeric.notna().any() else None,
                    "max": float(numeric.max()) if numeric.notna().any() else None,
                    "mean": float(numeric.mean()) if numeric.notna().any() else None,
                    "median": float(numeric.median()) if numeric.notna().any() else None,
                    "std": float(numeric.std()) if numeric.notna().any() else None,
                    "zero_count": int((numeric == 0).sum()),
                    "negative_count": int((numeric < 0).sum()),
                    "positive_count": int((numeric > 0).sum()),
                })

                if column_type == "Numerical | Discrete":
                    column_meta["value_counts_top_20"] = (
                        numeric.value_counts(dropna=True)
                        .head(20)
                        .to_dict()
                    )

            # -------------------------
            # Datetime
            # -------------------------
            elif column_type.startswith("Datetime"):
                dt = pd.to_datetime(s, errors="coerce")

                column_meta.update({
                    "min_date": str(dt.min()) if dt.notna().any() else None,
                    "max_date": str(dt.max()) if dt.notna().any() else None,
                    "date_range_days": (
                        int((dt.max() - dt.min()).days)
                        if dt.notna().any()
                        else None
                    ),
                })

                if dt.notna().any():
                    column_meta.update({
                        "year_count": int(dt.dt.year.nunique()),
                        "month_count": int(dt.dt.month.nunique()),
                        "day_count": int(dt.dt.day.nunique()),
                    })

            # -------------------------
            # Categorical
            # -------------------------
            elif column_type.startswith("Categorical"):
                value_counts = s.value_counts(dropna=True)

                column_meta.update({
                    "category_count": int(s.nunique(dropna=True)),
                    "top_categories": value_counts.head(20).to_dict(),
                    "most_frequent": (
                        value_counts.index[0]
                        if len(value_counts) > 0
                        else None
                    ),
                    "most_frequent_count": (
                        int(value_counts.iloc[0])
                        if len(value_counts) > 0
                        else 0
                    ),
                })

                if column_type == "Categorical | Binary":
                    column_meta["binary_values"] = (
                        s.dropna().unique().tolist()
                    )

            # -------------------------
            # Text
            # -------------------------
            elif column_type.startswith("Text"):
                text = s.dropna().astype(str)

                if len(text) > 0:
                    lengths = text.str.len()

                    column_meta.update({
                        "avg_length": float(lengths.mean()),
                        "min_length": int(lengths.min()),
                        "max_length": int(lengths.max()),
                        "median_length": float(lengths.median()),
                    })

                column_meta["sample_values"] = text.head(2).tolist()

            # -------------------------
            # Email
            # -------------------------
            elif column_type == "Email":
                text = s.dropna().astype(str)

                column_meta.update({
                    "sample_values": text.head(5).tolist(),
                    "unique_emails": text.nunique(),
                    "unique_domains": int(
                        text.str.extract(r"@(.+)$", expand=False)
                        .nunique()
                    )
                })

            # -------------------------
            # Phone
            # -------------------------
            elif column_type == "Phone Number":
                text = s.dropna().astype(str)

                column_meta.update({
                    "sample_values": text.head(5).tolist(),
                    "avg_length": (
                        float(text.str.len().mean())
                        if len(text) > 0
                        else None
                    ),
                    "unique_numbers": text.nunique()
                })

            # -------------------------
            # URL
            # -------------------------
            elif column_type == "URL":
                text = s.dropna().astype(str)

                column_meta["sample_values"] = text.head(5).tolist()
                column_meta["unique_urls"] = text.nunique()
            # -------------------------
            # Geographic
            # -------------------------
            elif column_type.startswith("Geographic"):
                text = s.dropna()

                column_meta["sample_values"] = text.head(10).tolist()

                if column_type == "Geographic | Latitude":
                    numeric = pd.to_numeric(s, errors="coerce")

                    column_meta.update({
                        "min": float(numeric.min()) if numeric.notna().any() else None,
                        "max": float(numeric.max()) if numeric.notna().any() else None,
                        "mean": float(numeric.mean()) if numeric.notna().any() else None,
                        "valid_range": [-90, 90],
                        "out_of_range": int(
                            ((numeric < -90) | (numeric > 90)).sum()
                        )
                    })

                elif column_type == "Geographic | Longitude":
                    numeric = pd.to_numeric(s, errors="coerce")

                    column_meta.update({
                        "min": float(numeric.min()) if numeric.notna().any() else None,
                        "max": float(numeric.max()) if numeric.notna().any() else None,
                        "mean": float(numeric.mean()) if numeric.notna().any() else None,
                        "valid_range": [-180, 180],
                        "out_of_range": int(
                            ((numeric < -180) | (numeric > 180)).sum()
                        )
                    })

            # -------------------------
            # Financial / Percentage
            # -------------------------
            elif (column_type == "Currency") or (column_type == "Percentage"):
                numeric = pd.to_numeric(s, errors="coerce")

                column_meta.update({
                    "min": float(numeric.min()) if numeric.notna().any() else None,
                    "max": float(numeric.max()) if numeric.notna().any() else None,
                    "mean": float(numeric.mean()) if numeric.notna().any() else None,
                    "median": float(numeric.median()) if numeric.notna().any() else None,
                })

            # -------------------------
            # File / Media
            # -------------------------
            elif column_type.startswith(("File Path", "Image Path", "Audio Path", "Video Path")):
                column_meta["sample_values"] = (
                    s.dropna().astype(str).head(5).tolist()
                )

            # -------------------------
            # Missing / Unknown
            # -------------------------
            elif column_type in ["Unknown","Empty","Missing"]:
                column_meta["sample_values"] = (
                    s.dropna().head(5).tolist()
                )

            # -------------------------
            # Targets
            # -------------------------
            elif column_type.startswith("Target"):
                column_meta["sample_values"] = (
                    s.dropna().head(10).tolist()
                )

                if "Regression" in column_type:
                    numeric = pd.to_numeric(s, errors="coerce")

                    column_meta.update({
                        "min": float(numeric.min()) if numeric.notna().any() else None,
                        "max": float(numeric.max()) if numeric.notna().any() else None,
                        "mean": float(numeric.mean()) if numeric.notna().any() else None,
                        "std": float(numeric.std()) if numeric.notna().any() else None,
                    })

                elif "Classification" in column_type:
                    column_meta["class_count"] = int(
                        s.nunique(dropna=True)
                    )
                    column_meta["class_distribution"] = (
                        s.value_counts(dropna=True).to_dict()
                    )
                    column_meta["missing_values"] = (
                        s.isna().sum()
                    )

            metadata[column] = column_meta

        self.column_metadata = metadata
        return metadata