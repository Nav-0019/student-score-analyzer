import pandas as pd
import numpy as np
import os

os.chdir("B:/Github/student-score-analyzer")
df = pd.read_csv("Data/ResearchInformation3.csv")
print(df.head())