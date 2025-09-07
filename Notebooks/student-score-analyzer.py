import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir("B:/Github/student-score-analyzer")

df = pd.read_csv("Data/ResearchInformation3.csv")

Department = pd.get_dummies(df['Department'], prefix='Department', drop_first=True)
df = pd.concat([df, Department], axis=1)

df['Gender'] = df['Gender'].map({'Male': 0, 'Female':1})

df['SSC'] = pd.cut( df['SSC'], bins=[0, 4.4, 4.8, 5.0], labels=[1, 2, 3], include_lowest=True, right = True).astype(int)
df['HSC'] = pd.cut( df['HSC'], bins=[0, 3.5, 4.2, 5.0], labels=[1, 2, 3], include_lowest=True, right = True).astype(int)

df['Income'] = df['Income'].str.strip()
income_map = {
    'Low (Below 15,000)': 1,
    'Lower middle (15,000-30,000)': 2,
    'Upper middle (30,000-50,000)': 3,
    'High (Above 50,000)': 4
}
df['Income'] = df['Income'].map(income_map)

df['Hometown'] = df['Hometown'].map({'Village':True, 'City':False})

df['Preparation'] = df['Preparation'].map({'More than 3 Hours':3, '0-1 Hour':1, '2-3 Hours':2})

df['Gaming'] = df['Gaming'].map({'0-1 Hour':1, 'More than 3 Hours':3, '2-3 Hours':2})

df['Attendance'] = df['Attendance'].map({'80%-100%':4, 'Below 40%':1, '60%-79%':3, '40%-59%':2})

df['Job'] = df['Job'].map({'Yes': True, 'No': False})

df['Extra'] = df['Extra'].map({'Yes': True, 'No': False})

Sem = pd.get_dummies(df['Semester'], prefix='Semester', drop_first=True)
df = pd.concat([df, Sem], axis=1)
df.drop(columns=['Semester', 'Department'], inplace=True)

x = df.drop(['Last', 'Overall'], axis= 1)
y = df[['Last', 'Overall']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

Y_pred = model.predict(x_test)

mse_last = mean_squared_error(y_test['Last'], Y_pred[:,0])
mse_overall = mean_squared_error(y_test['Overall'], Y_pred[:,1])

r2_last = r2_score(y_test['Last'], Y_pred[:,0])
r2_overall = r2_score(y_test['Overall'], Y_pred[:,1])

print("Last Semester - MSE:", mse_last, "R²:", r2_last)
print("Overall GPA   - MSE:", mse_overall, "R²:", r2_overall)
