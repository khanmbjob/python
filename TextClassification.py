import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
data = pd.read_csv('Pandas.csv', header = 0)
data.head()

# select text and author column from data 
col = ['Project Name', 'Remarks', 'Weight']
df = data[col]
print (df)


# assign numbers to authors [0,1,2]
df['category_id'] = df['Remarks'].factorize()[0]
category_id_df = df[['Remarks', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Remarks']].values)
df.head()

