# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:28:28 2019

@author: lenovo
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import collections
#classification algo
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection  import train_test_split


plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('bank-additional-full.csv', delimiter=";", header='infer')

#understanding the basics of bank data
data.describe()
data.columns
data.info()
data.head()
data.shape
#checking for missing values
data.isnull().sum() #surprisingly not a single null data

#dropping the duplicates
data = data.drop_duplicates()

#change yes->1 & no->0 
data['y'] = (data['y']=='yes').astype(float)
print(data['y'].value_counts())

#grouping education
data['education'].unique()
data['education']=np.where(data['education']=='basic.4y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.6y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.9y','Basic',data['education'])

#DAta Exploration
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')

count_nosub=len(data[data['y']==0])
count_sub=len(data[data['y']==1])
tot_sub = count_nosub+count_sub
pct_nosub = count_nosub/tot_sub
pct_sub = count_sub/tot_sub
print("percentage of no subscription = ",pct_nosub*100)
print("percentage of subscription = ",pct_sub*100)

print("Skew: %.2f" %data['y'].skew())

#we want to see how yes and no affects other numerical features
data.groupby('y').mean()

#categorical var
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

#viz

ax = sns.boxplot(x="y", y='duration',data=data)

ax = sns.boxplot(x="y", y='age',data=data)

pd.crosstab(data.job, data.y).plot(kind='bar')
plt.title('Purchase frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.education, data.y).plot(kind='bar')
plt.title('Purchase frequency for levels of education')
plt.xlabel('education')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.marital, data.y).plot(kind='bar')
plt.title('Purchase frequency for Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Purchase frequency for Day of week')
plt.xlabel('Day of week')
plt.ylabel('Frequency for purchase')


pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Purchase frequency for month')
plt.xlabel('month')
plt.ylabel('Frequency for purchase')
 

pd.crosstab(data.poutcome, data.y).plot(kind='bar')
plt.title('Purchase frequency for poutcome')
plt.xlabel('poutcome')
plt.ylabel('Frequency for purchase')



#DUMMY VARIABLE
cat_val = ['job','marital','education','default',
            'housing','loan','contact','month','day_of_week','poutcome']
for var in cat_val:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_val=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_val]

data_final=data[to_keep]
data_final.columns.values

#We noticed the imbalance in our data (yes or no)

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled ",len(os_data_X))
print("# of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("# of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#RFE
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#we get the selected features
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

#implement model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#removing the columns with high p-value

#####   WHOM TO MAKE A CALL ###########
cols=['euribor3m', 'job_blue-collar', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.
      format(logreg.score(X_test, y_test)))

#confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))


#Before adjusting 
print("The number of customers to be reached:", len(y_train))
print("The num of respondents are:",y_train.value_counts())
print("Cost associated for reaching out to a customer: ",len(y_train)*10)
print("Revenue: ", y_train.value_counts()[1]*50)
print("Profit(revenue-cost): ",y_train.value_counts()[1]*50-len(y_train)*10)

#after Model
print("The number of customers to be reached:", len(y_pred))
mod_yPred = collections.Counter(y_pred)
print("The number of respondents are: ", mod_yPred)
print("Cost associated with reaching out a customer: ",len(y_pred)*10)
print("Revenue: ", mod_yPred[1]*50)
print("Profit(revenue-cost): ",mod_yPred[1]*50-len(y_pred)*10)

print("profit percentage before model: ",(y_train.value_counts()[1]*50-len(y)*10)/(y.value_counts()[1]*50)*100)
print("profit percentage after model: ",(mod_yPred[1]*50-len(y_pred)*10)/(mod_yPred[1]*50)*100)

