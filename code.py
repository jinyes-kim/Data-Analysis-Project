# 코드 정리하기
"""
솔루션이랑 합치기
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Attrition 
# True == 1 False == 0

# load dataset
HR = pd.read_csv("C:/project-IBM/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
HR.index.name = 'record'
HR.info()
HR.describe()

# drop one data type columns 
HR.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'],
        axis="columns", inplace=True)


print(HR.columns, len(HR.columns))



# =====================================================
# Visualization
# =====================================================

# numerical columns
num_col = []
for i, col in enumerate(HR.columns):
    if HR[col].dtype != object:
        num_col.append(col)
        print("[{}]".format(i+1), end=' ')
        print(col, HR[col].unique())
        print()
        
        
# categorical colmuns 
cate_col = []
for i, col in enumerate(HR.columns):
    if HR[col].dtype == object:
        cate_col.append(col)
        print("[{}]".format(i+1), end=' ')
        print(col, HR[col].unique())
        print()
        

print(HR)



# 종속 변수의 개수가 다르므로 밀도를 이용해서 비교
sns.countplot(HR['Attrition'])

# label encoding
enc = preprocessing.LabelEncoder()
HR["Attrition"] = enc.fit_transform(HR.Attrition)
num_col.append('Attrition')
print(num_col)

"""
# numeric datas
plt.figure(figsize=(40, 40))
for i, column in enumerate(num_col, 1):
    plt.subplot(6, 4, i)
    HR[HR["Attrition"] == 0][column].hist(bins=35, color='green', label='Attrition = NO',alpha=0.5, density=True)
    HR[HR["Attrition"] == 1][column].hist(bins=35, color='red', label='Attrition = YES',alpha=0.5, density=True)
    plt.legend()
    plt.xlabel(column)
# categorical data
plt.figure(figsize=(40, 40))
for i, column in enumerate(cate_col, 1):
    plt.subplot(4, 2, i)
    HR[HR["Attrition"] == 0][column].hist(bins=35, color='green', label='Attrition = NO',alpha=0.5, density=True)
    HR[HR["Attrition"] == 1][column].hist(bins=35, color='red', label='Attrition = YES',alpha=0.5, density=True)
    plt.legend()
    plt.xlabel(column)
plt.tight_layout()
"""


# =====================================================
# tranform categorical data to dummy
# =====================================================

# dummy variable for categorical features
marital_dummy = pd.get_dummies(HR['MaritalStatus'])
overTime_dummy =  pd.get_dummies(HR['OverTime'])
role_dummy = pd.get_dummies(HR['JobRole'])
busi_dummy = pd.get_dummies(HR['BusinessTravel'])
depart_dummy = pd.get_dummies(HR['Department'])
edu_dummy = pd.get_dummies(HR['EducationField'])
gender_dummy = pd.get_dummies(HR['Gender'])

print(marital_dummy)
print(overTime_dummy)


for col in HR.columns:
    if HR[col].dtype != object or col == 'Attrition':
        continue
    else:
        HR.drop(col, axis="columns", inplace=True)
    
    
print(HR.columns, len(HR.columns))


HR = HR.join(marital_dummy.add_prefix('Marital_'))
HR = HR.join(overTime_dummy.add_prefix('OverTime_'))
HR = HR.join(role_dummy.add_prefix('jobRole_'))
HR = HR.join(busi_dummy.add_prefix('BusinessTrabel_'))
HR = HR.join(depart_dummy.add_prefix('department_'))
HR = HR.join(edu_dummy.add_prefix('EduField_'))
HR = HR.join(gender_dummy.add_prefix('Gender_'))
print(HR.columns, len(HR.columns))




# =====================================================
# Corrleation
# =====================================================

# find correlation
cor = HR.corr() # 상관계수


# visualize with Seaborn heat map
plt.figure(figsize=(40, 30))
sns.heatmap(abs(cor), annot = False, cmap = plt.cm.Blues)
plt.tight_layout()
plt.show()


# get correlation values with Attrition variable
cor_target = cor["Attrition"]
print(cor_target)
cor_target = abs(cor["Attrition"]) 



# 상관계수와 전체 변수로 돌린 모델의 p-value로 변수 셀렉션
# choose features above threshold 0.16

selected_cols = cor_target[cor_target > 0.10]
print("correlation > 0.16")
print(selected_cols)
selected_cols = selected_cols.drop('OverTime_No')
selected_cols.index



# selected features
HR_sel = HR[selected_cols.index]
print("\n\n*select features")
print(HR_sel.columns)

    


# =====================================================
# logistic regression function
# =====================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
import statsmodels.api as sm


def lrm(data, msg):
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
       	   
    scaler = StandardScaler()
    
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_std, columns = X.columns)
    
    model = LogisticRegression()
    model.fit(X_train_std, y_train)
    
    print("\n\n\t\t<Logistic Regression Score - " + msg +">\n\n\n")
    y_pred = model.predict(X_test_std)
    
    print(classification_report(y_test, y_pred))
       
    # report
    res = sm.Logit(y, X_std)
    result = res.fit()
    print(result.summary())



# =====================================================
# logistic regression report - All features
# =====================================================
lrm(HR, 'All features')
 


# =====================================================
# logistic regression report - selecing features
# =====================================================

#### top 5 featrues
lrm(HR_sel, '5 features')
 


# =====================================================
# optimize
# drop - TotalWorkingYears
# add -EnvironmentSatisfaction
# =====================================================

# 1차 드롭
selected_cols = selected_cols.drop('MonthlyIncome')
selected_cols = selected_cols.drop('TotalWorkingYears')
selected_cols = selected_cols.drop('JobLevel')
selected_cols = selected_cols.drop('StockOptionLevel')
selected_cols = selected_cols.drop('BusinessTrabel_Travel_Frequently')
selected_cols = selected_cols.drop('YearsWithCurrManager')


# 변수 추가
selected_cols['DistanceFromHome'] = HR['DistanceFromHome']




result = HR[selected_cols.index]
lrm(result, 'optimize')
