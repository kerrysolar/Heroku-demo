#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# For randomized data splitting
from sklearn.model_selection import train_test_split

# To build linear regression_model
import statsmodels.api as sm
# To check model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle 


# In[ ]:





# In[46]:


df = pd.read_csv('/Users/haoyuechang/Desktop/auto-mpg.csv')
#cData = pd.read_csv("auto-mpg.csv")


# In[47]:


#Data processing
df.info()
#398 entries, 9 columns


# In[48]:


#drop car name since it is not useful to analyze
df1 = df.drop(["car name"], axis=1)


# In[63]:


hpIsDigit = pd.DataFrame(
    df1.horsepower.str.isdigit()
)  # if the string is made of digits store True else False

# print the entries where isdigit = False
df1[hpIsDigit["horsepower"] == False]


# In[65]:


df1= df1.replace("?", np.nan)
df1[hpIsDigit["horsepower"] == False]


# In[66]:


df1.median()


# In[67]:


# Let's replace the missing values with median values of the columns.
# Note that we do not need to specify the column names below.
# Every column's missing value is replaced with that column's median respectively

medianFiller = lambda x: x.fillna(x.median())
df1 = df1.apply(medianFiller, axis=0)


# In[68]:


# let's convert the horsepower column from object type to float type
df1["horsepower"] = df1["horsepower"].astype(float)


# In[70]:


#Bivariate
df_attr = df1.iloc[:, 0:7]
sns.pairplot(
    df_attr, diag_kind="kde")


# In[71]:


# drop_first=True will drop one of the three origin columns
#vreat dummy variables
df2 = pd.get_dummies(df1, columns=["origin"], drop_first=True)
df2.head()


# In[72]:


#SPLIT DATA
# independent variables
X = df2.drop(["mpg"], axis=1)
# dependent variable
y = df2[["mpg"]]


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1)


# In[74]:


print(X_train.head())


# In[75]:


print(X_test.head())


# In[76]:


olsmod = sm.OLS(y_train, X_train)
olsres = olsmod.fit()


# In[77]:


print(olsres.summary())


# In[1]:


import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='a car mileage would be  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




