#!/usr/bin/env python
# coding: utf-8

# ## Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve


# ## Data Preprocessing

# In[2]:


pd.set_option('display.max_columns',90)
data=pd.read_excel(r'C:\Users\ASUS\Downloads\default of credit card clients.xls')
data


# In[3]:


data.describe(include='all')


# In[4]:


data.drop('ID', axis=1, inplace=True)


# In[5]:


data.isnull().sum()


# In[6]:


data.shape


# In[7]:


data.corr()['default']


# In[8]:


avarage_corr=data.corr()['default'].mean()


# In[9]:


avarage_corr


# In[10]:


data.dtypes


# In[11]:


data.columns


# In[12]:


dropped_columns = []

for i in data[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default']]:
    
    if abs(data.corr()['default'][i])<avarage_corr:
        dropped_columns.append(i)
    
data.drop(dropped_columns, axis=1, inplace=True)  


# In[13]:


data


# In[14]:


data.columns


# In[15]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[[
    'LIMIT_BAL', 
    'PAY_0', 
    'PAY_2', 
    'PAY_3',
    'PAY_4', 
    'PAY_5', 
    'PAY_6'
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# In[16]:


for i in data[[
    'LIMIT_BAL', 
    'PAY_0', 
    'PAY_2', 
    'PAY_3',
    'PAY_4', 
    'PAY_5', 
    'PAY_6']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[17]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[18]:


for i in data[[
    'LIMIT_BAL', 
    'PAY_0', 
    'PAY_2', 
    'PAY_3',
    'PAY_4', 
    'PAY_5', 
    'PAY_6']]:
    
    
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])


# In[19]:


for i in data[[
    'LIMIT_BAL', 
    'PAY_0', 
    'PAY_2', 
    'PAY_3',
    'PAY_4', 
    'PAY_5', 
    'PAY_6']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[20]:


data=data.reset_index(drop=True)


# In[21]:


data.describe(include='all')


# ## WOE Transformation for Logistic Regression

# In[22]:


data.head()


# In[23]:


new_data=data.copy()


# In[24]:


new_data.head()


# In[25]:


grouped=new_data.groupby(['SEX', 'default'])['default'].count().unstack().reset_index()
grouped


# In[26]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['SEX_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[27]:


new_data=new_data.merge(grouped[['SEX', 'SEX_woe']], how='left', on='SEX')
new_data


# In[28]:


grouped=new_data.groupby(['EDUCATION', 'default'])['default'].count().unstack().reset_index()
grouped


# In[29]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['EDUCATION_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[30]:


new_data=new_data.merge(grouped[['EDUCATION', 'EDUCATION_woe']], how='left', on='EDUCATION')
new_data


# In[31]:


grouped=new_data.groupby(['MARRIAGE', 'default'])['default'].count().unstack().reset_index()
grouped


# In[32]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['MARRIAGE_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[33]:


new_data=new_data.merge(grouped[['MARRIAGE', 'MARRIAGE_woe']], how='left', on='MARRIAGE')
new_data


# In[34]:


ranges=[-np.inf, new_data['LIMIT_BAL'].quantile(0.25), new_data['LIMIT_BAL'].quantile(0.5), new_data['LIMIT_BAL'].quantile(0.75), np.inf]
new_data['LIMIT_BAL_category']=pd.cut(new_data['LIMIT_BAL'], bins=ranges)
new_data


# In[35]:


grouped=new_data.groupby(['LIMIT_BAL_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[36]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['LIMIT_BAL_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[37]:


new_data=new_data.merge(grouped[['LIMIT_BAL_category', 'LIMIT_BAL_woe']], how='left', on='LIMIT_BAL_category')
new_data


# In[38]:


ranges=[-np.inf, new_data['PAY_0'].quantile(0.25), new_data['PAY_0'].quantile(0.5), new_data['PAY_0'].quantile(0.75), np.inf]
new_data['PAY_0_category']=pd.cut(new_data['PAY_0'], bins=ranges,  duplicates='drop') #Error message said that only unique values must be added, so, I dropped duplicates.
new_data


# In[39]:


grouped=new_data.groupby(['PAY_0_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[40]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_0_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[41]:


new_data=new_data.merge(grouped[['PAY_0_category', 'PAY_0_woe']], how='left', on='PAY_0_category')
new_data


# In[42]:


ranges=[-np.inf, new_data['PAY_2'].quantile(0.25), new_data['PAY_2'].quantile(0.5), new_data['PAY_2'].quantile(0.75), np.inf]
new_data['PAY_2_category']=pd.cut(new_data['PAY_2'], bins=ranges, duplicates='drop') 
new_data


# In[43]:


grouped=new_data.groupby(['PAY_2_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[44]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_2_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[45]:


new_data=new_data.merge(grouped[['PAY_2_category', 'PAY_2_woe']], how='left', on='PAY_2_category')
new_data


# In[46]:


ranges=[-np.inf, new_data['PAY_3'].quantile(0.25), new_data['PAY_3'].quantile(0.5), new_data['PAY_3'].quantile(0.75), np.inf]
new_data['PAY_3_category']=pd.cut(new_data['PAY_3'], bins=ranges, duplicates='drop') 
new_data


# In[47]:


grouped=new_data.groupby(['PAY_3_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[48]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_3_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[49]:


new_data=new_data.merge(grouped[['PAY_3_category', 'PAY_3_woe']], how='left', on='PAY_3_category')
new_data


# In[50]:


ranges=[-np.inf, new_data['PAY_4'].quantile(0.25), new_data['PAY_4'].quantile(0.5), new_data['PAY_4'].quantile(0.75), np.inf]
new_data['PAY_4_category']=pd.cut(new_data['PAY_4'], bins=ranges, duplicates='drop') 
new_data


# In[51]:


grouped=new_data.groupby(['PAY_4_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[52]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_4_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[53]:


new_data=new_data.merge(grouped[['PAY_4_category', 'PAY_4_woe']], how='left', on='PAY_4_category')
new_data


# In[54]:


ranges=[-np.inf, new_data['PAY_5'].quantile(0.25), new_data['PAY_5'].quantile(0.5), new_data['PAY_5'].quantile(0.75), np.inf]
new_data['PAY_5_category']=pd.cut(new_data['PAY_5'], bins=ranges, duplicates='drop') 
new_data


# In[55]:


grouped=new_data.groupby(['PAY_5_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[56]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_5_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[57]:


new_data=new_data.merge(grouped[['PAY_5_category', 'PAY_5_woe']], how='left', on='PAY_5_category')
new_data


# In[58]:


ranges=[-np.inf, new_data['PAY_6'].quantile(0.25), new_data['PAY_6'].quantile(0.5), new_data['PAY_6'].quantile(0.75), np.inf]
new_data['PAY_6_category']=pd.cut(new_data['PAY_6'], bins=ranges, duplicates='drop') 
new_data


# In[59]:


grouped=new_data.groupby(['PAY_6_category', 'default'])['default'].count().unstack().reset_index()
grouped


# In[60]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PAY_6_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[61]:


new_data=new_data.merge(grouped[['PAY_6_category', 'PAY_6_woe']], how='left', on='PAY_6_category')
new_data


# In[62]:


new_data.columns


# In[63]:


woe_data=new_data[[
    'SEX_woe',
    'EDUCATION_woe',
    'MARRIAGE_woe', 
    'LIMIT_BAL_woe',
    'PAY_0_woe',
    'PAY_2_woe',
    'PAY_3_woe',
    'PAY_4_woe',
    'PAY_5_woe',
    'PAY_6_woe',
    'default'
]]


# In[64]:


woe_data


# In[65]:


woe_data.isnull().sum()


# In[66]:


woe_data['EDUCATION_woe']=woe_data['EDUCATION_woe'].fillna(value=woe_data['EDUCATION_woe'].mean())


# In[67]:


woe_data.isnull().sum()


# In[68]:


woe_data.shape


# ## Modeling

# In[69]:


X=woe_data.drop('default', axis=1)
y=woe_data['default']


# In[70]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[71]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred_test)
    report=classification_report(y_test, y_pred_test)
    
    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Confusion Matrix', confusion_matrix)
    
    print('Classification report', report)


# In[72]:


lr=LogisticRegression()


# In[73]:


lr.fit(X_train, y_train)


# In[74]:


result_lr=evaluate(lr, X_test, y_test)


# In[75]:


y_prob = lr.predict_proba(X_test)[:, 1]

roc_prob= roc_auc_score(y_test, y_prob)
gini = 2*roc_prob-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_prob)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# In[76]:


lr_balanced=LogisticRegression(class_weight='balanced')
lr_balanced.fit(X_train, y_train)


# In[77]:


result_lr_balanced=evaluate(lr_balanced, X_test, y_test)


# In[78]:


y_prob = lr_balanced.predict_proba(X_test)[:, 1]

roc_prob= roc_auc_score(y_test, y_prob)
gini = 2*roc_prob-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_prob)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# ## Modeling for other ML algorithms

# In[79]:


data.head()


# In[80]:


data_dummied=pd.get_dummies(data, drop_first=True)


# In[81]:


data_dummied


# In[82]:


data_dummied.columns


# In[83]:


data_dummied=data_dummied[['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'SEX_male', 'EDUCATION_high school',
       'EDUCATION_not educated', 'EDUCATION_others', 'EDUCATION_university',
       'MARRIAGE_others', 'MARRIAGE_single','default']]


# In[84]:


data_dummied


# In[85]:


X=data_dummied.drop('default', axis=1)
y=data_dummied['default']


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[87]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred_test)
    report=classification_report(y_test, y_pred_test)
    
    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Confusion Matrix', confusion_matrix)
    
    print('Classification report', report)


# In[88]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[89]:


result_dtc=evaluate(dtc, X_test, y_test)


# In[90]:


y_prob = dtc.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[91]:


rfc_base=RandomForestClassifier()
rfc_base.fit(X_train, y_train)


# In[92]:


result_rfc_base=evaluate(rfc_base, X_test, y_test)


# In[93]:


y_prob = rfc_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[94]:


from sklearn.feature_selection import SelectFromModel


# In[95]:


sfm = SelectFromModel(rfc_base)
sfm.fit(X_train, y_train)


# In[96]:


selected_feature= X_train.columns[(sfm.get_support())]
selected_feature


# In[97]:


feature_scores = pd.Series(rfc_base.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores


# In[98]:


X_train=X_train[['LIMIT_BAL', 'PAY_0', 'PAY_2']]
X_test=X_test[['LIMIT_BAL', 'PAY_0', 'PAY_2']]


# In[99]:


rfc_importance=RandomForestClassifier()
rfc_importance.fit(X_train, y_train)


# In[100]:


result_rfc_importance=evaluate(rfc_importance, X_test, y_test)


# In[101]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[102]:


rfc_randomized = RandomizedSearchCV(estimator = rfc_base, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=1, random_state=42, n_jobs = -1)

rfc_randomized.fit(X_train, y_train)


# In[103]:


result_rfc_randomized=evaluate(rfc_randomized, X_test, y_test)


# In[104]:


y_prob = rfc_importance.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[105]:


y_prob = rfc_randomized.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[106]:


catboost_base_dummy=CatBoostClassifier()
catboost_base_dummy.fit(X_train, y_train)


# In[107]:


result_catboost_base_dummy=evaluate(catboost_base_dummy, X_test, y_test)


# In[108]:


y_prob = catboost_base_dummy.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[109]:


xgboost_base=XGBClassifier()
xgboost_base.fit(X_train, y_train)


# In[110]:


result_xgboost_base=evaluate(xgboost_base, X_test, y_test)


# In[111]:


y_prob = xgboost_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[112]:


lightgbm_base=LGBMClassifier()
lightgbm_base.fit(X_train, y_train)


# In[113]:


result_lightgbm_base=evaluate(lightgbm_base, X_test, y_test)


# In[114]:


y_prob = lightgbm_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[115]:


#Hyperparameter Tuning (Lightgbm)
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]    
}

param_distributions


# In[116]:


lightgbm_randomized=RandomizedSearchCV(lightgbm_base, 
                                       param_distributions=param_distributions, 
                                       n_iter=10, cv=5, 
                                       n_jobs=-1, 
                                       random_state=42)

lightgbm_randomized.fit(X_train, y_train)


# In[117]:


print('Best hyperparameters for the lightgbm:', lightgbm_randomized.best_params_)


# In[118]:


optimized_lightgbm=lightgbm_randomized.best_estimator_


# In[119]:


result_optimized_lightgbm = evaluate(optimized_lightgbm, X_test, y_test)


# In[120]:


y_prob = optimized_lightgbm.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[121]:


#Hyperparameter Tuning (XGBoost)

param_distributions = {
    
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'subsample': np.linspace(0.5, 1, num=6),
    'colsample_bytree': np.linspace(0.5, 1, num=6),
    'gamma': [0,1,5,10]
    
}

param_distributions


# In[122]:


xgboost_randomized = RandomizedSearchCV(xgboost_base, 
                                        param_distributions=param_distributions, 
                                        n_iter=10, cv=5, 
                                        n_jobs=-1, 
                                        random_state=42)
xgboost_randomized.fit(X_train, y_train)


# In[123]:


print('Best hyperparameters for XGBoost:', xgboost_randomized.best_params_)


# In[124]:


optimized_xgboost=xgboost_randomized.best_estimator_


# In[125]:


result_optimized_xgboost=evaluate(optimized_xgboost, X_test, y_test)


# In[126]:


y_prob = optimized_xgboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[127]:


#Hyperparameter Tuning (CatBoost)

param_distributions = {
    
    'iterations': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': np.linspace(2, 30, num=7)
    
}

param_distributions


# In[128]:


catboost_randomized=RandomizedSearchCV(catboost_base_dummy, 
                                       param_distributions=param_distributions, 
                                       cv=5, n_iter=10, 
                                       random_state=42)

catboost_randomized.fit(X_train, y_train)


# In[129]:


print('Best hyperparameters for CatBoost:', catboost_randomized.best_params_)


# In[130]:


optimized_catboost=catboost_randomized.best_estimator_


# In[131]:


result_optimized_catboost=evaluate(optimized_catboost, X_test, y_test)


# In[132]:


y_prob = optimized_catboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[133]:


#Stacking Model

base_classifiers = [
    catboost_base_dummy,
    optimized_xgboost,
    optimized_lightgbm,
    rfc_randomized
    
]


# In[134]:


meta_classifier = optimized_catboost


# In[135]:


stacking_classifier = StackingCVClassifier(classifiers=base_classifiers,
                                           meta_classifier=meta_classifier,
                                           cv=5,
                                           use_probas=True,
                                           use_features_in_secondary=True,
                                           verbose=1,
                                           random_state=42)


# In[136]:


stacking_classifier.fit(X_train, y_train)


# In[137]:


result_stacking_classifier=evaluate(stacking_classifier, X_test, y_test)


# In[138]:


y_prob = stacking_classifier.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[139]:


svc_base=SVC(probability=True)
svc_base.fit(X_train, y_train)


# In[140]:


result_svc_base=evaluate(svc_base, X_test, y_test)


# In[141]:


y_prob = svc_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[142]:


from sklearn.model_selection import RandomizedSearchCV

kernel = ['linear', 'poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto'] 

C = [1, 10, 100, 1000]


random_grid = {'kernel': kernel,
               'gamma': gamma,
               'C': C}
print(random_grid)


# In[143]:


svc_randomized = RandomizedSearchCV(estimator = svc_base, 
                                    param_distributions = random_grid, 
                                    n_iter = 1, 
                                    cv = 2, 
                                    verbose=1, 
                                    n_jobs = -1)

svc_randomized.fit(X_train, y_train)


# In[144]:


svc_randomized.best_params_


# In[145]:


optimized_svc=svc_randomized.best_estimator_
result_optimized_svc=evaluate(optimized_svc, X_test, y_test)


# In[146]:


y_prob = svc_randomized.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[147]:


optimized_svc_2=SVC(kernel='rbf', gamma='auto', C=1000, probability=True)


# In[148]:


optimized_svc_2.fit(X_train, y_train)


# In[149]:


result_optimized_svc_2=evaluate(optimized_svc_2, X_test, y_test)


# In[150]:


y_prob = optimized_svc_2.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Univariate Analysis

# In[151]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    optimized_catboost.fit(X_train_single, y_train)
    y_prob_train_single=optimized_catboost.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    optimized_catboost.fit(X_test_single, y_test)
    y_prob_test_single=optimized_catboost.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   


# ## Catboost model with categorical columns

# In[152]:


data.head()


# In[153]:


X=data.drop('default', axis=1)
y=data['default']


# In[154]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[155]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred_test)
    report=classification_report(y_test, y_pred_test)
    
    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Confusion Matrix', confusion_matrix)
    
    print('Classification report', report)


# In[156]:


catboost_with_cat=CatBoostClassifier(cat_features=['SEX', 'EDUCATION', 'MARRIAGE'])
catboost_with_cat.fit(X_train, y_train)


# In[157]:


result_catboost_with_cat=evaluate(catboost_with_cat, X_test, y_test)


# In[158]:


y_prob = catboost_with_cat.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[197]:


from catboost import CatBoostClassifier, Pool
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

variables = []
train_Gini = []
test_Gini = []


catboost_with_cat = CatBoostClassifier()

for col in X.columns:

    X_train_single = X_train[[col]]
    X_test_single = X_test[[col]]

    cat_features_indices = [X_train_single.columns.get_loc(c) for c in categorical_cols if c in X_train_single]
    train_pool = Pool(X_train_single, label=y_train, cat_features=cat_features_indices)
    test_pool = Pool(X_test_single, label=y_test, cat_features=cat_features_indices)
    

    catboost_with_cat.fit(train_pool, verbose=False)
    

    y_prob_train_single = catboost_with_cat.predict_proba(train_pool)[:, 1]
    roc_prob_train = roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train = 2 * roc_prob_train - 1
    

    y_prob_test_single = catboost_with_cat.predict_proba(test_pool)[:, 1]
    roc_prob_test = roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test = 2 * roc_prob_test - 1
    

    variables.append(col)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)


df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})


df = df.sort_values(by='Test Gini', ascending=False)


df


# In[ ]:




