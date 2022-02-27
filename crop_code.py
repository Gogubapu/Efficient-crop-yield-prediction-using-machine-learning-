import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve,  accuracy_score
from sklearn import ensemble
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from pandas import DataFrame
from sklearn.metrics import r2_score


from sklearn.model_selection import cross_val_score
from sklearn import metrics as ms

import warnings
warnings.filterwarnings("ignore")

##load data
crop_data = pd.read_csv(r'cropdata.csv')
crop_data.head()

soil_data = pd.read_csv(r'soildata.csv')

rainfall_data = pd.read_csv(r'raindata.csv')

Data=pd.concat([crop_data,soil_data,rainfall_data],join='inner',axis=1)
merge_Data=DataFrame(Data)
print(merge_Data.head())
##preprocessing
merge_Data.isnull().sum()

merge_Data=merge_Data.dropna(axis=0,how='any',inplace=False)

merge_Data.isna().any()
merge_Data.dtypes

##label encoder

merge_Data_En=merge_Data

le=LabelEncoder()

# Iterating over all object data type common columns
for col in merge_Data.columns.values:
       # Encoding only categorical variables
    if merge_Data[col].dtypes=='object':
        # Converting object data type to category
        merge_Data[col]=merge_Data[col].astype('str')
       # Using whole data to form an exhaustive list of levels
        data=merge_Data[col]
        le.fit(data.values)
        merge_Data_En[col]=le.transform(merge_Data[col])

        merge_Data.head()
        merge_Data.dtypes

y = merge_Data_En.level
X = merge_Data_En.drop('level', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% train and 20% test

# feature extraction
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
print(model.feature_importances_)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
##classification

##knn
n_neighbors=5
knn=neighbors.KNeighborsClassifier(n_neighbors,weights='uniform')
knn.fit(X_train,y_train)
y_knn=knn.predict(X_test)
knn_score=accuracy_score(y_test,y_knn)
print("predicted knn:",y_knn)
print("knn:",knn_score)

##decision tree
Dtree = tree.DecisionTreeClassifier()
Dtree.fit(X_train, y_train)
y_pre = Dtree.predict(X_test)
Dtree_score=accuracy_score(y_test,y_pre)
print("predicted tree:",y_pre)
print("tree:",Dtree_score)

##svm
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pre = svm_rbf.predict(X_test)
svm_rbf_score=accuracy_score(y_test,y_pre)
print("predicted svm:",y_pre)
print("svm_rbf:",svm_rbf_score)
'''
##svm_linear
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pre = svm_linear.predict(X_test)
svm_linear_score=accuracy_score(y_test,y_pre)
print("svm_linear:",svm_linear_score)
'''


accuracy=[knn_score,Dtree_score,svm_rbf_score]
col={'Accuracy':accuracy}
models=['Knn','Decision Tree','svm_rbf']
df=DataFrame(data=col,index=models)
df

df.plot(kind='bar')
plt.show()

##regression

y = merge_Data_En.level
X = merge_Data_En.drop('Production', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% train and 20% test


from sklearn.metrics import mean_squared_error
##knn
n_neighbors=5
knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
knn.fit(X_train,y_train)
y_knn=knn.predict(X_test)
rms_knn=np.sqrt(mean_squared_error(y_test, y_knn))
print("rms_knn",rms_knn)

#print("mse:",mean_squared_error(y_test,y_knn))
##decision tree
Dtree = tree.DecisionTreeRegressor()
Dtree.fit(X_train, y_train)
y_pre = Dtree.predict(X_test)
rms_dtree = np.sqrt(mean_squared_error(y_test, y_pre))
print("rms_dtree",rms_dtree)

##svm
svm_rbf = svm.SVR(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pre = svm_rbf.predict(X_test)
rms_svm = np.sqrt(mean_squared_error(y_test, y_pre))
print("rms_sr",rms_svm)



mean_squared_error=[rms_knn,rms_dtree,rms_svm]
col={'MSE':mean_squared_error}
models=['Knn','Decision Tree','svm_rbf']
df=DataFrame(data=col,index=models)
print(df)

df.plot(kind='bar')
plt.show()
##cross validation
knn = KNeighborsRegressor()
knn_cv=cross_val_score(knn, X_train, y_train, cv=10)
print('knn_cv',knn_cv)
svm_rbf = svm.SVR()
svm_rbf_cv=cross_val_score(svm_rbf, X_train, y_train, cv=10)
print('svm_rbf',svm_rbf_cv)
d_tree=tree.DecisionTreeRegressor()
tree_cv=cross_val_score(d_tree, X_train, y_train, cv=10)
print('decision tree',tree_cv)

plt.plot( knn_cv)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()

plt.plot(svm_rbf_cv)
plt.xlabel('svm_rbf')
plt.ylabel('Cross-validated accuracy')
plt.show()


plt.plot(tree_cv)
plt.xlabel('decision tree')
plt.ylabel('Cross-validated accuracy')
plt.show()

