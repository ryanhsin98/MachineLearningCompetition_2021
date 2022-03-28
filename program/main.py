#!/usr/bin/env python
# coding: utf-8

"""
Created on Sun Jun 20 16:38:50 2021

@author: Team_16
"""

#%%
# =============================================================================
# 前期準備
# =============================================================================
'''
置入所需資料處理與繪圖套件
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import display


'''
匯入資料
'''
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_data = df_train.append(df_test)

#重新索引
df_data = df_data.reset_index()


#%%
# =============================================================================
# 遺漏值處理
# =============================================================================

'''
遺漏值個數
'''
#Pclass(133), Age(189), Cabin(690) 缺失值較多
for col in df_data.columns.tolist():          
    print('{} column missing values: {}'.format(col, df_data[col].isnull().sum()))


'''
Fare補值
'''
#查看遺漏值乘客的資訊
df_data[df_data['Fare'].isnull()]

#因具遺漏值的乘客為三等艙，故補三等艙、SibSp=0、Parch=0的fare中位數
med_fare = df_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df_data['Fare'] = df_data['Fare'].fillna(med_fare)


'''
Pclass補值
'''

#補值第一步驟，按ticket補值，同樣ticket會是同樣艙等：
deplicate_ticket = []
for ticket in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == ticket, 'Fare']
    if tem.count() > 1:
        deplicate_ticket.append(df_data.loc[df_data.Ticket == ticket,['Name','Ticket','Fare','Pclass','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)

#一個ticket對一個pclass
deplicate_ticket_dropna = deplicate_ticket.dropna() #先刪除Pclass是NA的人
match_tp = deplicate_ticket_dropna.drop_duplicates(subset=['Ticket'], keep='first', inplace=False)
match_tp = pd.concat((match_tp['Ticket'],match_tp['Pclass']),axis = 1)
match_tp_dict = match_tp.set_index('Ticket')['Pclass'].to_dict()

#按ticket上的艙等補值
df_data.Pclass = df_data.Pclass.fillna(df_data.Ticket.map(match_tp_dict))

#補值第二步驟，按fare補值：
#觀察Pclass與fare的彼此分布狀況
f, ax = plt.subplots(figsize=(8,3))
ax.set_title('Pclass Fare dist', size=20)
sns.distplot(df_data[df_data.Pclass == 1].dropna().Fare, hist=False, color='black', label='P-1')
sns.distplot(df_data[df_data.Pclass == 2].dropna().Fare, hist=False, color='green', label='P-2')
sns.distplot(df_data[df_data.Pclass == 3].dropna().Fare, hist=False, color='blue', label='P-3')
sns.distplot(df_data[df_data.Pclass.isnull() == True].Fare, hist=False, color='red', label='P-NA')
ax.legend(fontsize=15)

#觀察Fare在各艙等的四分位線狀況
print(df_data[df_data.Pclass==2].Fare.describe())
print(df_data[df_data.Pclass==3].Fare.describe())
print(df_data[df_data.Pclass==1].Fare.describe())
#觀察結論：75%的三艙等fare低於15.5；75%的二艙等fare低於26

#補值條件：Fare低於15.5為三艙等，Fare低於26為二艙等，其他為一艙等
#取出pclass不為na的dataframe
no_na = df_data.loc[df_data['Pclass'].isnull() == False]
#取出pclass為na的dataframe
yes_na = df_data.loc[df_data['Pclass'].isnull() == True]

fill_p1 = yes_na.loc[df_data['Fare'] > 26]
fill_p2 = yes_na.loc[(df_data['Fare'] >15.5) & (df_data['Fare'] <= 26)]
fill_p3 = yes_na.loc[df_data['Fare'] <= 15.5]

p1 = fill_p1.fillna({'Pclass':1})
p2 = fill_p2.fillna({'Pclass':2})
p3 = fill_p3.fillna({'Pclass':3})

df_data = pd.concat([no_na, p1, p2, p3])


'''
Age補值
'''
#Age和乘客name中的title有關，創造新特徵-Title
df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })

#各Title的age中位數
age_med = df_data.groupby('Title')['Age'].median().values
df_data['New_Age'] = df_data['Age']

#補值Age：按title的中位數補值成新特徵 - New_Age
for i in range(0,5):
    df_data.loc[(df_data.Age.isnull()) & (df_data.Title == i),'New_Age'] = age_med[i]
df_data['New_Age'] = df_data['New_Age'].astype('int')


#%%
# =============================================================================
# 觀察現有特徵
# =============================================================================

'''
Pclass觀察
'''
#結論：Pclass與生存率有關，可採納進模型
y_dead = df_data[df_data.Survived==0].groupby('Pclass')['Survived'].count()
y_alive = df_data[df_data.Survived==1].groupby('Pclass')['Survived'].count()
pos = [1, 2, 3]
ax = plt.figure(figsize=(8,4)).add_subplot(111)
ax.bar(pos, y_dead, color='r', alpha=0.6, label='dead')
ax.bar(pos, y_alive, color='g', bottom=y_dead, alpha=0.6, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticks(pos)
ax.set_xticklabels(['Pclass%d'%(i) for i in range(1,4)], size=15)
ax.set_title('Pclass Surveved count', size=20)

display(df_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().round(3))

'''
SibSp觀察
'''
#結論：SibSp與生存率有關，可採納進模型
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
df_data.groupby('SibSp')['Survived'].mean().plot(kind='bar', ax=ax1)
ax1.set_title('SibSp Survival Rate', size=16)
ax1.set_xlabel('')

display(df_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().round(3))


#%%
# =============================================================================
# 新特徵建立
# =============================================================================

'''
新特徵： Pclass與性別的交叉類別變數 - female-p1, female-p3
'''

#觀察Pclass交叉性別下的存活率狀況
#結論一：同性別下，等級越高，存活率越高
#結論二：死亡率方面，女性與男性皆在三艙最高

label = []
for sex_i in ['female','male']:
    for pclass_i in range(1,4):
        label.append('sex:%s,Pclass:%d'%(sex_i, pclass_i))

pos = range(6)
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(111)
ax.bar(pos, 
    df_data[df_data['Survived']==0].groupby(['Sex','Pclass'])['Survived'].count().values, 
    color='r',alpha=0.6,
    tick_label=label, 
    label='dead')
ax.bar(pos, 
    df_data[df_data['Survived']==1].groupby(['Sex','Pclass'])['Survived'].count().values, 
    bottom=df_data[df_data['Survived']==0].groupby(['Sex','Pclass'])['Survived'].count().values,
    color='g',alpha=0.6,
    tick_label=label,
    label='alive')
ax.tick_params(labelsize=10)
ax.set_title('Sex_Pclass_Survived', size=20)
ax.legend(fontsize=15,loc='best')

display(df_data[['Sex', 'Pclass','Survived']].groupby(['Sex','Pclass'], as_index=False).mean().round(3))

#創造新特徵：Pclass與性別交叉分群
p1 = df_data.loc[df_data['Pclass'] == 1]
p2 = df_data.loc[df_data['Pclass'] == 2]
p3 = df_data.loc[df_data['Pclass'] == 3]

p1['female-p1'] = p1['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
p1['male-p1'] = p1['Sex'].map({'female' : 0, 'male' : 1}).astype('int')
p1['female-p2'] = p1['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p1['male-p2'] = p1['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p1['female-p3'] = p1['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p1['male-p3'] = p1['Sex'].map({'female' : 0, 'male' : 0}).astype('int')

p2['female-p1'] = p2['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p2['male-p1'] = p2['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p2['female-p2'] = p2['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
p2['male-p2'] = p2['Sex'].map({'female' : 0, 'male' : 1}).astype('int')
p2['female-p3'] = p2['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p2['male-p3'] = p2['Sex'].map({'female' : 0, 'male' : 0}).astype('int')

p3['female-p1'] = p3['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p3['male-p1'] = p3['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p3['female-p2'] = p3['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p3['male-p2'] = p3['Sex'].map({'female' : 0, 'male' : 0}).astype('int')
p3['female-p3'] = p3['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
p3['male-p3'] = p3['Sex'].map({'female' : 0, 'male' : 1}).astype('int')

#合併處理後的data
df_data = pd.concat([p1, p2, p3])


'''
新特徵 - Family Size
'''
#Family_size特徵建立
df_data['Family_size'] = df_data['SibSp'] + df_data['Parch'] + 1

#觀察Family_size與存活率間的關係，結論為有相關
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(211)
sns.barplot(x=df_data['Family_size'].value_counts().index, y=df_data['Family_size'].value_counts().values, ax=ax1)
ax1.set_title('Family Size Feature Value Counts', size=10)
ax1.set_xlabel('')

ax2 = fig.add_subplot(212)
df_data.groupby(df_data.Family_size)['Survived'].mean().plot(kind='bar', ax=ax2)
ax2.set_title('Family_Size Survival Rate', size=10)

'''
新特徵 - Age_Code_4
'''
#觀察New_Age與存活率間的關係
fig, axes = plt.subplots(2,1,figsize=(8,6))
sns.set_style('white')
sns.distplot(df_data.New_Age, rug=True, color='b', ax=axes[0])
ax0 = axes[0]
ax0.set_title('Age distribution')
ax0.set_xlabel('')

ax1 = axes[1]
ax1.set_title('Age survival distribution')
k1 = sns.distplot(df_data[df_data.Survived==0].New_Age, hist=False, color='r', ax=ax1, label='dead')
k2 = sns.distplot(df_data[df_data.Survived==1].New_Age, hist=False, color='g', ax=ax1, label='alive')
ax1.set_xlabel('')
ax1.legend(fontsize=16)

#New_Age 離散化(分四箱)
df_data['Age_Code_4'] = pd.qcut(df_data['New_Age'], 4).cat.codes
display(df_data[['Age_Code_4', 'Survived']].groupby(['Age_Code_4'], as_index=False).mean().round(3))

'''
新特徵 - Connected_survival
'''
#同樣票號的乘客群，很有可能會一起死亡或一起存活
deplicate_ticket = []
for tk in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == tk, 'Fare']
    if tem.count() > 1:
        deplicate_ticket.append(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Family_size','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print('friends: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size == 1]))
print('families: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size > 1]))

#default 
df_data['Connected_Survival'] = 0.5 
for _, df_grp in df_data.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows():
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 0

#觀察Connected_Survival與存活率間的關係
df_data.groupby('Connected_Survival')[['Survived']].agg({np.mean, 'count'}).round(3)


#%%
# =============================================================================
# 整理要用於模型中的數據
# =============================================================================

#重新按PassengerId排順序
df_data = df_data.sort_values(by=['PassengerId'])

#拆開出train和test_data
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

#logistic數據整理
logi_select = df_train.drop(['Cabin','Age','Name','Sex','Ticket','Embarked','PassengerId'],axis=1)
Selected_features = ['Title', 'Pclass', 'SibSp','female-p1', 'female-p3', 'Family_size','Connected_Survival','Age_Code_4']

#分離出feature與target的數據
train_x = logi_select[Selected_features]
train_y = logi_select['Survived']
test_x = df_test[Selected_features]

#整理數據成模型所適合採用的array型態
training_data = pd.concat([train_y,train_x],axis=1) 
test_data = pd.concat([df_test['Survived'], df_test[Selected_features]],axis=1)
class_of_training_data = np.array(train_y)
train_feature = np.array(train_x)
test_feature  = np.array(df_test[Selected_features])


#%%
# =============================================================================
# 建立logistic模型
# =============================================================================

def initial_parameter(train_feature):
    return(np.random.randn(1,1),np.random.randn(train_feature.shape[1],1))

#logistic模型為sigmoid函數
def sigmoid(train_feature,theta0_initial_val,theta_initial_val):
    power=theta0_initial_val+np.dot(train_feature,theta_initial_val)
    return(1/(1+ np.exp(-power)))

#theta0微分Cost Function，並求最小化的Cost Function
def deri_wrt_theta0(temp,train_feature):
    return((np.sum(temp))/train_feature.shape[0])

#theta微分Cost Function(加上L2 Regularization)，並求最小化的Cost Function
def deri_wrt_theta(temp,train_feature,lam,theta_initial):
    return(((np.matmul((train_feature.T),temp))/train_feature.shape[0]) -
           ((lam/train_feature.shape[0])*theta_initial))

def cost_function(class_of_training_data,H_theta0_theta_A,theta_A,train_feature,lam):
    a=np.matmul((class_of_training_data.T),np.log(H_theta0_theta_A))
    b=np.matmul(((1-class_of_training_data).T),np.log(1-H_theta0_theta_A))
    c=(a+b)/train_feature.shape[0]
    d=(lam*np.sum(np.square(theta_A)))/2*train_feature.shape[0]
    return(-c+d)
    
class_of_training_data=np.reshape(class_of_training_data,(training_data.shape[0],1))

#梯度下降法，求模型參數
def gradient_descent(alpha,epsilon,train_feature,class_of_training_data,lam):
    #parameter's
    #Alpha(Learning Rate)
    #Epsilon : to stop gradient descent at proper minimized state
    #lam(lambda) : For L2 Regularisation

    theta0_initial,theta_initial=initial_parameter(train_feature)
    
    i=0
    iterations=[]
    neg_log_like_loss = []
    
    while(True):
        H_theta0_theta_i=sigmoid(train_feature,theta0_initial,theta_initial)
        Tem =H_theta0_theta_i-class_of_training_data
        Derivative_theta0 =deri_wrt_theta0(Tem,train_feature)
        Derivative_thetas =deri_wrt_theta(Tem,train_feature,lam,theta_initial)
        neg_log_like_initial= cost_function(class_of_training_data,H_theta0_theta_i,theta_initial,train_feature,lam)      
        
        #更新參數值
        theta0_final = theta0_initial - alpha*Derivative_theta0
        theta_final = theta_initial - alpha*Derivative_thetas
    
        H_theta0_theta_f = sigmoid(train_feature,theta0_final,theta_final)
        neg_log_like_final= cost_function(class_of_training_data,H_theta0_theta_f,theta_final,train_feature,lam)
        
        if abs(neg_log_like_initial - neg_log_like_final) < epsilon:
            return(theta0_final,theta_final,iterations,neg_log_like_loss)
            break
            
        theta0_initial = theta0_final
        theta_initial = theta_final
    
        i += 1
    
        iterations.append(i)
        neg_log_like_loss.append(neg_log_like_initial)
        

#%%
# =============================================================================
# 訓練模型與預測
# =============================================================================

'''
訓練模型
'''
theta0_final,theta_final,iterations,neg_log_like_loss=gradient_descent(0.01,0.0000005
                                                                       ,train_feature,class_of_training_data,0.0000001)

#輸出模型參數：theta0_final,theta_final
print(theta0_final)
print(theta_final)

plt.scatter(iterations,neg_log_like_loss,data="Logistic Regression")
plt.title('Minimization Curve of Cost Function in Logistic Regression')


'''
預測測試集的存活資料
'''
def predict(test_feature,theta0_final,theta_final):
    output_list=[]
    sigmoid_function_value=sigmoid(test_feature,theta0_final,theta_final)    
    for i in sigmoid_function_value:
        if i>=0.50:
            output_list.append(1)
        else:
            output_list.append(0)
    return (output_list)
   
output_list = predict(test_feature,theta0_final,theta_final)


'''
輸出測試資料成csv
'''
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':output_list})
submission.to_csv("Team_16.csv", index=False)


