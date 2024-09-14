#!/usr/bin/env python
# coding: utf-8

# In[177]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[2]:


path1 = 'refinery/'
path2 = 'temperature/'
ref_files = os.listdir(path1)
temp_files = os.listdir(path2)


# In[3]:


months = ['فروردین','اردیبهشت','خرداد','تیر','مرداد','شهریور','مهر','آبان','آذر','دی','بهمن','اسفند']
d_temp = pd.DataFrame([])
for i in range(5):
    for name in months:
        d = pd.read_excel(os.path.join(path2,temp_files[i]),sheet_name=name)
        d = d[d.columns[[0,2]]]
        d_temp = pd.concat((d_temp,d))


# In[4]:


d_ref = pd.DataFrame([])
for i in range(5):
    for name in months:
        d = np.transpose(pd.read_excel(os.path.join(path1,ref_files[i]),sheet_name=name))
        d.columns = d.iloc[0]
        d.drop(index='Unnamed: 0',inplace=True)
        d_ref = pd.concat((d_ref,d))


# In[5]:


d_ref.index = np.arange(0,len(d_ref.index))
d_temp.index = d_ref.index
data = pd.concat((d_temp,d_ref),axis=1)
data.info()


# In[6]:


years = ['1398','1399','1400','1401','1402']
d_rain = pd.DataFrame([])
for year in years:
    d = np.transpose(pd.read_excel('آمار بارندگی روزانه.xlsx',sheet_name=year))
    d.drop(columns=d.columns[0:3],inplace=True)
    for col in d.columns:
        d_rain = pd.concat((d_rain,d[col][1:-1]))
ind = np.where(d_rain.isnull())[0]
d_rain.index = np.arange(0,d_rain.shape[0])
d_rain.drop(index=ind,inplace=True)


# In[7]:


d_rain.index = data.index
data = pd.concat((data,d_rain),axis=1)


# In[8]:


col_names = list(data.columns)
col_names[-1] = 'بارندگی'
data.columns = col_names


# In[9]:


data


# In[10]:


data.fillna(method='ffill',inplace=True)


# In[11]:


data[['دمای میانگین\n(ċ)','بارندگی']]


# In[12]:


fig,ax1 = plt.subplots(figsize=[12,5])
ax1.plot(data['بارندگی'],color='red',label='rainfall')
ax1.set_xlabel('days')
ax1.set_ylabel('rainfall(mm)')
ax2 = ax1.twinx()
ax2.plot(data['كدورت آب ورودي در 95 درصد شرايط (NTU)'],color='blue',label='turbidity')
ax2.set_ylabel('turbidity(NTU)')
fig.legend()


# In[13]:


fig,ax1 = plt.subplots(figsize=[12,5])
ax1.plot(data['دمای میانگین\n(ċ)'],color='red',label='temperature')
ax1.set_xlabel('days')
ax1.set_ylabel('n(ċ)')
ax2 = ax1.twinx()
ax2.plot(data['كدورت آب ورودي در 95 درصد شرايط (NTU)'],color='blue',label='turbidity')
ax2.set_ylabel('turbidity(NTU)')
fig.legend()


# In[180]:


print('پیش بینی کدورت آب ورودی')
X = data['بارندگی']
y = data['كدورت آب ورودي در 95 درصد شرايط (NTU)']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
X = data['بارندگی']
y = data['كل جامدات محلول (TDS)-آب خام ورودي (ميلي گرم بر ليتر)']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی مصرف مواد منعقد کننده آب ورودی')
X = data['بارندگی']
y = data['مصرف مواد منعقد كننده اوليه -خالص (كيلوگرم)']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[149]:


print('پیش بینی کل کدورت آب ورودی')
X = data[['دمای میانگین\n(ċ)','بارندگی','دبي آب ورودي تصفيه خانه (مترمكعب)']]
y = data['كدورت آب ورودي در 95 درصد شرايط (NTU)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor(n_estimators=100,max_depth=35,criterion='squared_error',random_state=100)
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor(n_estimators=200,learning_rate=0.2,max_depth=3,random_state=100)
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

# cross validation
RF_Regressor = RandomForestRegressor()
y_pred = cross_val_predict(RF_Regressor,X.to_numpy().reshape(-1,1),y)
print('\n\n Random forest with one feature (cross validation)')
print(f'mean squared value is: {mean_squared_error(y_pred,y)} \n')
print(f'r2 score is: {r2_score(y_pred,y)}')
GB_Regressor = GradientBoostingRegressor()
y_pred = cross_val_predict(GB_Regressor,X.to_numpy().reshape(-1,1),y)
print('\n\n Gradient Boosting with one feature (cross validation)')
print(f'mean squared value is: {mean_squared_error(y_pred,y)} \n')
print(f'r2 score is: {r2_score(y_pred,y)}')


print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
X = data[['دمای میانگین\n(ċ)','بارندگی','دبي آب ورودي تصفيه خانه (مترمكعب)']]
y = data['كل جامدات محلول (TDS)-آب خام ورودي (ميلي گرم بر ليتر)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor(n_estimators=100,max_depth=35,criterion='squared_error',random_state=100)
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor(n_estimators=200,learning_rate=0.2,max_depth=3,random_state=100)
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی مصرف مواد منعقد کننده آب ورودی')
X = data[['دمای میانگین\n(ċ)','بارندگی','دبي آب ورودي تصفيه خانه (مترمكعب)']]
y = data['مصرف مواد منعقد كننده اوليه -خالص (كيلوگرم)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor(n_estimators=100,max_depth=35,criterion='squared_error',random_state=100)
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor(n_estimators=200,learning_rate=0.2,max_depth=3,random_state=100)
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[41]:


d_temp_av = []
for i in range(5):
    for name in months:
        d = pd.read_excel(os.path.join(path2,temp_files[i]),sheet_name=name)
        d = d[d.columns[2]].mean()
        d_temp_av.append(d)
d_temp_av


# In[154]:


d_tur_av = []
d_tds_av = []
d_mat_av = []
d_debi_av = []
for i in range(5):
    for name in months:
        d = np.transpose(pd.read_excel(os.path.join(path1,ref_files[i]),sheet_name=name))
        d.columns = d.iloc[0]
        d.drop(index='Unnamed: 0',inplace=True)
        d_tur_av.append(d[d.columns[2]].mean())
        d_tds_av.append(d[d.columns[7]].mean())
        d_mat_av.append(d[d.columns[6]].mean())
        d_debi_av.append(d[d.columns[0]].mean())


# In[155]:


d_rain_av = []
for year in years:
    d = np.transpose(pd.read_excel('آمار بارندگی روزانه.xlsx',sheet_name=year))
    d.drop(columns=d.columns[0:3],inplace=True)
    for col in d.columns:
        d_rain_av.append(d[col][-1])


# In[156]:


data_av = pd.DataFrame()
data_av['کدورت متوسط'] = d_tur_av
data_av['دمای متوسط'] = d_temp_av
data_av['مجموع بارش'] = d_rain_av
data_av['دبی متوسط'] = d_debi_av
data_av['مواد منعقد کننده متوسط'] = d_mat_av
data_av['محلول در آب متوسط'] = d_tds_av


# In[162]:


data_av


# In[20]:


fig,ax1 = plt.subplots(figsize=[12,5])
ax1.plot(data_av['مجموع بارش'],color='red',label='rainfall')
ax1.set_xlabel('months')
ax1.set_ylabel('rainfall(mm)')
ax2 = ax1.twinx()
ax2.plot(data_av['کدورت متوسط'],color='blue',label='turbidity')
ax2.set_ylabel('turbidity(NTU)')
fig.legend()


# In[21]:


fig,ax1 = plt.subplots(figsize=[12,5])
ax1.plot(data_av['دمای متوسط'],color='red',label='temperature')
ax1.set_xlabel('months')
ax1.set_ylabel('rainfall(mm)')
ax2 = ax1.twinx()
ax2.plot(data_av['کدورت متوسط'],color='blue',label='turbidity')
ax2.set_ylabel('turbidity(NTU)')
fig.legend()


# In[159]:


print('پیش بینی کل جامدات محلول در آب ورودی')
X = data_av['مجموع بارش']
y = data_av['کدورت متوسط']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
X = data_av['مجموع بارش']
y = data_av['محلول در آب متوسط']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی مواد منقعد کننده در آب ورودی')
X = data_av['مجموع بارش']
y = data_av['مواد منعقد کننده متوسط']
v_n = [1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for monthly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[164]:


print('پیش بینی کدورت آب ورودی')
X = data_av[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_av['کدورت متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
X = data_av[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_av['محلول در آب متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


print('\n\n\n')
print('پیش بینی مواد منعقد کننده در آب ورودی')
X = data_av[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_av['مواد منعقد کننده متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for monthly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[174]:


data_weekly = pd.DataFrame([])
for i in range(data.shape[0]//7):
    d = data.iloc[1+7*i-1:(i+1)*7]
    t1 = d[d.columns[[1,2,4,8,9]]].mean()
    t2 = d[d.columns[-1]].sum()
    data_weekly = pd.concat((data_weekly,pd.concat((t1,pd.Series(t2)))),axis=1)
data_weekly = np.transpose(data_weekly)
data_weekly.columns = ['دمای متوسط','دبی متوسط','کدورت متوسط','مواد منعقد کننده متوسط','محلول در آب متوسط','مجموع بارش']


# In[175]:


data_weekly


# In[173]:


print('پیش بینی کدورت آب ورودی')
X = data_weekly['مجموع بارش']
y = data_weekly['کدورت متوسط']
v_n = [1000,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
X = data_weekly['مجموع بارش']
y = data_weekly['محلول در آب متوسط']
v_n = [1000,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی مواد منعقد کننده در آب ورودی')
X = data_weekly['مجموع بارش']
y = data_weekly['مواد منعقد کننده متوسط']
v_n = [1000,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train.to_numpy().reshape(-1,1))
scaler.transform(X_test.to_numpy().reshape(-1,1))
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = neuralNetRegressor.predict(X_test.to_numpy().reshape(-1,1))
print('Neural Net with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = RF_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Random forest with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train.to_numpy().reshape(-1,1),y_train)
y_pred = GB_Regressor.predict(X_test.to_numpy().reshape(-1,1))
print('\n\n Gradient boosting with one feature for weekly data')
print(f'mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[176]:


print('پیش بینی کدورت آب ورودی')
v_n = [1000]
X = data_weekly[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_weekly['کدورت متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی کل جامدات محلول در آب ورودی')
v_n = [1000]
X = data_weekly[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_weekly['محلول در آب متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')

print('\n\n\n')
print('پیش بینی مواد منعقد کننده در آب ورودی')
v_n = [1000]
X = data_weekly[['دبی متوسط','دمای متوسط','مجموع بارش']]
y = data_weekly['مواد منعقد کننده متوسط']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
neuralNetRegressor = MLPRegressor(v_n)
neuralNetRegressor.fit(X_train,y_train)
y_pred = neuralNetRegressor.predict(X_test)
print('\n\n Neural net with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
RF_Regressor = RandomForestRegressor()
RF_Regressor.fit(X_train,y_train)
y_pred = RF_Regressor.predict(X_test)
print('\n\n Random forest with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')
GB_Regressor = GradientBoostingRegressor()
GB_Regressor.fit(X_train,y_train)
y_pred = GB_Regressor.predict(X_test)
print('\n\n Gradient boosting with 3 features for weekly data')
print(f'\n \n mean squared value is: {mean_squared_error(y_pred,y_test)} \n')
print(f'r2 score is: {r2_score(y_pred,y_test)}')


# In[ ]:




