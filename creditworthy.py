
# coding: utf-8

#%% 1. IMPORT & FORMAT DATA
import pandas as pd

df = pd.read_csv('Training50.csv',sep=',')
tst = pd.read_csv('Test50.csv',sep=',')

#Save the credit worthiness column as a separate numpy array
y_train = df['Creditability'].values
df = df.drop(['Creditability','Unnamed: 0','Foreign.Worker','Telephone'],1)

y_tst = tst['Creditability'].values
tst = tst.drop(['Creditability','Unnamed: 0','Foreign.Worker','Telephone'],1)

# create three dummy variables using get_dummies, then exclude the first dummy column
bal_dum = pd.get_dummies(df['Account.Balance'], prefix='Bal',drop_first=True)
bal_dumt = pd.get_dummies(tst['Account.Balance'], prefix='Bal',drop_first=True)

pay_dum = pd.get_dummies(df['Payment.Status.of.Previous.Credit'], prefix='Payment',drop_first=True)
pay_dumt = pd.get_dummies(tst['Payment.Status.of.Previous.Credit'], prefix='Payment',drop_first=True)

pur_dum = pd.get_dummies(df['Purpose'], prefix='Purpose',drop_first=True)
pur_dumt = pd.get_dummies(tst['Purpose'], prefix='Purpose',drop_first=True)

mar_dum = pd.get_dummies(df['Sex...Marital.Status'], prefix='Sex',drop_first=True)
mar_dumt = pd.get_dummies(tst['Sex...Marital.Status'], prefix='Sex',drop_first=True)

aset_dum = pd.get_dummies(df['Most.valuable.available.asset'], prefix='asset',drop_first=True)
aset_dumt = pd.get_dummies(tst['Most.valuable.available.asset'], prefix='asset',drop_first=True)

apt_dum = pd.get_dummies(df['Type.of.apartment'], prefix='typeApt',drop_first=True)
apt_dumt = pd.get_dummies(tst['Type.of.apartment'], prefix='typeApt',drop_first=True)


# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
dfa = pd.concat([df, bal_dum, pay_dum,pur_dum,mar_dum,aset_dum,apt_dum], axis=1)
tsta = pd.concat([tst, bal_dumt, pay_dumt,pur_dumt,mar_dumt,aset_dumt,apt_dumt], axis=1)


feature_cols = ['Bal_2','Bal_3','Credit.Amount','Duration.of.Credit..month.','Value.Savings.Stocks','Length.of.current.employment','Instalment.per.cent',
               'Guarantors','Payment_2','Payment_3','Purpose_2','Purpose_3','Purpose_4','Sex_2','Sex_3','asset_2','asset_3','asset_4','typeApt_2','typeApt_3']

df1 = dfa[feature_cols]
tst1 = tsta[feature_cols]


x_train1 = df1.as_matrix()
x_tst1 = tst1.as_matrix()


# In[10]:

def random_forest(forest):
    forest.fit(x_train1, y_train)
    forest.score(x_train1, y_train)

    #print(forest.feature_importances_)
    #plt.plot(forest.feature_importances_)
    importance = list(df1)

    #imp_values = [(importance[x],forest.feature_importances_[x]) for x in range(1,len(importance))]
    #imp_values = sorted(imp_values,key=lambda x: x[1], reverse=True)

    imp = pd.DataFrame(forest.feature_importances_, importance, ['Score'])
    imp = imp.sort(['Score'], ascending=[0])
    #print(imp)

    #importance = list(my_dataframe.columns.values)

    P5 = forest.predict(x_tst1)
    MAE5 = abs(P5-y_tst)
    MAE5 = 1.0/500*sum(MAE5)
    return MAE5


#%% 2. LINEAR REGRESSION
import time

from sklearn import linear_model
from numpy import array 
from numpy import sum

time_start_reg = time.clock()

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train1, y_train)
linear.score(x_train1, y_train)

#Equation coefficient and Intercept
#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)

#Predict Linear Output P1
P1 = linear.predict(x_tst1)
P1 = array([1 if a>=0.5 else 0 for a in P1])

#Mean Absolute Error 
MAE1 = abs(P1-y_tst)
MAE1 = 1.0/500*sum(MAE1)
print(MAE1)
time_regr = (time.clock() - time_start_reg)

# In[13]:



from sklearn import tree
#import pydot
#from sklearn import StringIO
time_start_DT = time.clock()

model = tree.DecisionTreeClassifier(criterion='entropy') # for classification, 
# here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# PK: entropy gives better results for this dataset -> use it!
# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score
model.fit(x_train1, y_train)
model.score(x_train1, y_train)

#Predict Output
P2= model.predict(x_tst1)
MAE2 = abs(P2-y_tst)
MAE2 = 1.0/500*sum(MAE2)
print(MAE2)

time_DT = (time.clock() - time_start_DT)

# In[14]:



#Try to get figure!
#from IPython.display import Image 
#dot_data = StringIO() 
#tree.export_graphviz(model, out_file = dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png()) 

#%% SUPPORT VECTOR MACHINE

from sklearn import svm

time_start_svm = time.clock()

# Create SVM classification object 
vector = svm.SVC(kernel='linear') 

# Train the model using the training sets and check score
vector.fit(x_train1, y_train)
vector.score(x_train1, y_train)

#Predict Output
P3= vector.predict(x_tst1)
MAE3 = abs(P3-y_tst)
MAE3 = 1.0/500*sum(MAE3)
print(MAE3)
time_svm = (time.clock() - time_start_svm)


#%% NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
time_start_NB = time.clock()

bayes = GaussianNB() 
# Train the model using the training sets and check score
bayes.fit(x_train1, y_train)
bayes.score(x_train1, y_train)
#Predict Output
P4= bayes.predict(x_tst1)
MAE4 = abs(P4-y_tst)
MAE4 = 1.0/500*sum(MAE4)
print(MAE4)
time_NB = (time.clock() - time_start_NB)


#%% 3.  RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
#from seaborn import boxplot
from matplotlib import pyplot as plt

error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=5)   
    error += random_forest(forest)
    #print(MAE5)    
print('n_estimators 5: ', error/50.0)


error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=15)    
    error += random_forest(forest)
    #print(MAE5)    
print('n_estimators 15: ', error/50.0)


error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=20)    
    error += random_forest(forest)
    #print(MAE5)    
print('n_estimators 20: ', error/50.0)

#%%
from sklearn.ensemble import RandomForestClassifier

time_start_RF = time.clock()
error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features=20, criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=25)    
    error += random_forest(forest)
    #print(MAE5)    
print('n_estimators 25: ', error/50.0)
time_RF = (time.clock() - time_start_RF)

#%%
error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=40)    
    error += random_forest(forest)
    #print(MAE5)    
print('n_estimators 50: ', error/50.0)


error = 0
for i in range(1,50):
    forest = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=9, min_samples_split=9)    
    error += random_forest(forest)
    #print(MAE5)    
print('default n_estimators constraint: ', error/50.0)



# In[18]:

# 4. ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier

time_start_ERF = time.clock()
error = 0
for i in range(1,50):
    forest = ExtraTreesClassifier(max_features=20, criterion='entropy', max_depth=9, min_samples_split=9, n_estimators=25)    
    error += random_forest(forest)
    #print(MAE5)    
print('default n_estimators constraint: ', error/50.0)
time_ERF = (time.clock() - time_start_ERF)





