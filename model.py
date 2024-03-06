import pandas as pd 
pd.options.display.max_columns = 999
data = pd.read_csv('drug-overdose-death-rates new.csv')
d= data.drop(['Entity', 'Code', 'Any opioid death rates (CDC WONDER)',
       'Cocaine overdose death rates (CDC WONDER)',
       'Heroin overdose death rates (CDC WONDER)',
       'Synthetic opioids death rates (CDC WONDER)'], axis=1)
x = d[['Year']]
y = d['Prescription Opioids death rates (US CDC WONDER)']

model_1 = data.poly1d(data.polyfit(x, y, 1))
model_2 = data.poly1d(data.polyfit(x, y, 2))
model_3 = data.poly1d(data.polyfit(x, y, 3))
model_4 = data.poly1d(data.polyfit(x, y, 4))
model_5 = data.poly1d(data.polyfit(x, y, 5)) 