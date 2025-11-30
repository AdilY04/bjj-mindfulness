import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('datasets/ajwa+or+medjool/AjwaOrMejdool.csv', sep=';')
#print(df.head(5))
#print(df.columns)
d = {'Black': 0, 'Brown': 1}
df['Color'] = df['Color'].map(d)

features = ['Date Length (cm)', 'Date Diameter (cm)', 'Date Weight (g)',
            'Pit Length (cm)', 'Calories (Kcal)', 'Color']
x = df[features]
y = df['Class (Ajwa or Medjool)']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)
tree.plot_tree(dtree, feature_names=features)
plt.show()
