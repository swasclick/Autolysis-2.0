from flask import Flask
from parsers.tabular_data_parser import LLMTabularParser
import pandas as pd
import pprint

df = pd.read_csv(r"D:\Placement Prep\Projects\MCQ Solver\MCQ-Solver\dataset\test.csv")
parser = LLMTabularParser()
grps = parser.get_column_groups(df)

# print(grps)

isvalid, outs = parser.validate_column_classification(classification = grps,df = df)
print(isvalid)
# print(outs)
pprint.pprint(parser.get_metadata(df=df,classification=outs["corrected_classification"]))

# print(df.shape)