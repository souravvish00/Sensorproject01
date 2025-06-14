import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt


file_path = r"A:\VS Code\Sensor_Project\notebooks\wafer_23012020_041211.csv"
wafers = pd.read_csv(file_path)
print("shape of the feature store dataset:" , wafers.shape)
wafers.head()

wafers.rename(columns={'Unnamed : 0' : 'wafer'}, inplace= True)

wafers.drop(columns=['Good/Bad']).iloc[:100].to_csv('test.csv', index=False)


from sklearn.model_selection import train_test_split

wafers, wafers_test = train_test_split(wafers, test_size = 20, random_state = 42)

wafers.info()