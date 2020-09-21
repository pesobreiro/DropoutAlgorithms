# Adapted from https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
from sklearn.metrics import roc_auc_score
from math import sqrt
def roc_auc_ci(y_true, y_score, positive=1):
    '''
    # y_true = TRUE data
    # y_pred = predicted data at the model
    '''
    auc = roc_auc_score(y_true, y_score)
    n1 = sum(y_true == positive)
    n2 = sum(y_true get_ipython().getoutput("= positive)")
    q1 = auc / (2 - auc)
    q2 = 2*auc**2 / (1 + auc)
    se_auc = sqrt((auc*(1 - auc) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1*n2))
    lower = auc - 1.96*se_auc
    upper = auc + 1.96*se_auc
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)


import numpy as np
import pandas as pd
import time
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
import 


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# Disable warnings
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn import metrics
from sklearn.inspection import plot_partial_dependence
import time

import graphviz
import pydotplus
#import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image  


from sklearn.model_selection import GridSearchCV


from sklearn import tree
from dtreeviz.trees import *


get_ipython().run_line_magic("matplotlib", " inline")


#reading data
dt = pd.read_excel('dataset/fitnessCustomers.xlsx',index_col=0)


#rows and columns
dt.shape


#feature names
dt.columns


dt.head(2).T


# number Customers by dropout
dt.dropout.value_counts()


dt.head().T


# Calculate inscriptio month
dt['rmonth']=dt['startDate'].str.extract('-(\d\d)', expand=True)
dt['rmonth']=pd.to_numeric(dt['rmonth'])


# drop start date
dt.drop(axis=1,columns=['startDate'],inplace=True)


dt.columns


dt.head(2).T


dt.isnull().any()


dt.cfreq.value_counts()


# where are the nulls?
dt[pd.isnull(dt['cfreq'])]


# How many values are nulls?
dt['cfreq'][pd.isnull(dt['cfreq'])].shape


dt.shape


dt.dropna(axis=0,inplace=True)


dt.shape


dt.drop(axis=1,columns=['freeuse'],inplace=True)


dt.describe().T


dt.loc[dt.age<10].T


plt.rcParams['figure.figsize'] = [20, 5]
sns.countplot(x='age',data=dt,palette='rainbow').set_title('Customers by age')
plt.xticks(rotation=45);


dt=dt.loc[dt.ageget_ipython().getoutput("=0]")


dt['sex'].value_counts()


dt.describe().T


females, males = dt.sex.value_counts().ravel()
print('Females get_ipython().run_line_magic("d", " Males %d' %(females,males))")


dt['dayswfreq'][dt.dropout == 1].describe()


sns.countplot(x='dayswfreq',data=dt.loc[(dt.dropout == 1) & (dt.dayswfreq<100)],palette='rainbow').set_title('Days without frequencies - Dropout customers')
plt.xticks(rotation=45);


dt['dayswfreq'][dt.dropout == 0].describe()


sns.countplot(x='dayswfreq',data=dt.loc[(dt.dropout == 1) & (dt.dayswfreq<100)],palette='rainbow').set_title('Days without frequencies - non-dropout customers')
plt.xticks(rotation=45);


class_dropout=dt.dropout.value_counts()
class_dropout


dropout, non_dropout = dt.dropout.value_counts().ravel()
print('dropout get_ipython().run_line_magic(".2f%%", " non-dropout %.2f%%' %(dropout/(dropout+non_dropout)*100,non_dropout/(dropout+non_dropout)*100))")


# checking classes 
# dropout customer
print('Minority class represents get_ipython().run_line_magic(".2f%%'", " %(class_dropout[0]/(class_dropout[0]+class_dropout[1])*100))")


plt.rcParams['figure.figsize']=[17,12]
dt.hist();


X=dt.copy()
y=dt['dropout']
X.drop(axis=1,columns=['dropout'],inplace=True)


X.columns


X.info()


X.columns


X.describe().round(decimals=2)


y.describe().round(decimals=2)


plt.rcParams['figure.figsize'] = [12, 3]
X.age.hist(bins=20)


X.sex.value_counts()


# calculation class weights
n=len(y)
k=2
print(y.value_counts())
n0=np.bincount(y)[0]
n1=np.bincount(y)[1]
print('class 1 get_ipython().run_line_magic("2f", " class 2 %2f' % ((n/(2*n0)),(n/(k*n1))))")


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify = y, random_state=0)
print('Size X_train:',X_train.shape)
print('Size X_test:',X_test.shape)
print('Size y_train:',y_train.shape)
print('Size y_test:',y_test.shape)


print('base \tget_ipython().run_line_magic("s", " ratio %2f \ny_train %s ratio %2f \ny_test \t%s ratio %2f'")
      % (np.bincount(y), np.bincount(y)[0]/(np.bincount(y)[0]+np.bincount(y)[1]),
         np.bincount(y_train),np.bincount(y)[0]/(np.bincount(y)[0]+np.bincount(y)[1]),
         np.bincount(y_test), np.bincount(y)[0]/(np.bincount(y)[0]+np.bincount(y)[1])))


# getting params that can be optimized in gridsearch
LogisticRegression().get_params()


solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1', 'l2', 'elasticnet', 'none']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)


start_time = time.time()

scoring = {'AUC': 'roc_auc'}
gs = GridSearchCV(estimator = LogisticRegression(random_state=42, class_weight={0: 4.0, 1: 0.571}),
                  param_grid=grid,
                  scoring=scoring, 
                  refit='AUC', 
                  return_train_score=True)
grid_result = gs.fit(X_train,y_train)
print('best params',grid_result.best_params_)
print('best AUC score:',grid_result.best_score_)
print('execution time:',time.time()-start_time)


model_lr = LogisticRegression(C=1.0,penalty='l2', solver='liblinear' , random_state=42, class_weight={0: 4.0, 1: 0.571})
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
print(metrics.classification_report(y_test,y_pred_lr))
print(metrics.confusion_matrix(y_test,y_pred_lr))
#print("Features sorted by their score:") Logistic regression dont allow to sort features by their score
#print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), model.feature_importances_), X_train.columns),reverse=True))
matrix_LR=metrics.confusion_matrix(y_test, y_pred_lr)
print('AUC',metrics.roc_auc_score(y_test, y_pred_lr))
print('AUC_ROC',roc_auc_ci(y_test,y_pred_lr))


tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_lr).ravel()
print('tn:get_ipython().run_line_magic("d", " fp:%d fn:%d tp:%d' %(tn, fp, fn, tp))")
print('accuracy get_ipython().run_line_magic("f", " sensitivy %f, specificity %f, precision %f, f1score %f' \")
      get_ipython().run_line_magic("(((tp+tn)/(tp+tn+fn+fp)),(tp/(tp+fn)),tn/(tn+fp),tp/(tp+fp),", " 2*tp/(2*tp+fp+fn)))")


DecisionTreeClassifier().get_params()


max_features = ['auto', 'sqrt', 'log2','None']
max_depth = [3,4,5,6,7,8]
criterion = ['gini', 'entropy']
class_weight = ["balanced", "balanced_subsample"]

# define grid search
grid = dict(max_features=max_features,max_depth=max_depth, criterion=criterion, class_weight = class_weight)


start_time = time.time()

scoring = {'AUC': 'roc_auc'}
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state=42),
                  param_grid=grid,
                  scoring=scoring, 
                  refit='AUC', 
                  return_train_score=True)
grid_result = gs.fit(X_train,y_train)
print('best params',grid_result.best_params_)
print('best AUC score:',grid_result.best_score_)
print('execution time:',time.time()-start_time)


model_dtc = DecisionTreeClassifier(class_weight={0: 4.0, 1: 0.571}, criterion='entropy',max_depth = 6, max_features='auto', random_state=42)
model_dtc.fit(X_train,y_train)
y_pred_dtc = model_dtc.predict(X_test)
print(metrics.classification_report(y_test,y_pred_dtc))
print(metrics.confusion_matrix(y_test,y_pred_dtc))
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), model_dtc.feature_importances_), X_train.columns),reverse=True))
matrix_DTC=metrics.confusion_matrix(y_test, y_pred_dtc)
print('AUCROC:',metrics.roc_auc_score(y_test, y_pred_dtc))
print('AUC_ROC_CI',roc_auc_ci(y_test,y_pred_dtc))


tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_dtc).ravel()
print('tn:get_ipython().run_line_magic("d", " fp:%d fn:%d tp:%d' %(tn, fp, fn, tp))")
print('accuracy get_ipython().run_line_magic("f", " sensitivy %f, specificity %f, precision %f, f1score %f' \")
      get_ipython().run_line_magic("(((tp+tn)/(tp+tn+fn+fp)),(tp/(tp+fn)),tn/(tn+fp),tp/(tp+fp),", " 2*tp/(2*tp+fp+fn)))")


RandomForestClassifier().get_params()


n_estimators = [100,200, 500]
max_features = ['auto', 'sqrt', 'log2','None']
max_depth = [4,5,6,7,8]
criterion = ['gini', 'entropy']
class_weight = ["balanced", "balanced_subsample"]

# define grid search
grid = dict(n_estimators = n_estimators,max_features=max_features,max_depth=max_depth, criterion=criterion, class_weight = class_weight)


start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, stratify = y, random_state=0)

scoring = {'AUC': 'roc_auc'}
gs = GridSearchCV(estimator = RandomForestClassifier(random_state=42),
                  param_grid=grid,
                  scoring=scoring, 
                  refit='AUC', 
                  return_train_score=True)
grid_result = gs.fit(X_train,y_train)
print('best params',grid_result.best_params_)
print('best AUC score:',grid_result.best_score_)
print('execution time:',time.time()-start_time)


model_rfc = RandomForestClassifier(class_weight='balanced',criterion='gini',max_depth=8,max_features='auto',
                                   n_estimators=500, random_state=42)
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)
print(metrics.classification_report(y_test,y_pred_rfc))
print(metrics.confusion_matrix(y_test,y_pred_rfc))
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), model_rfc.feature_importances_), X_train.columns),reverse=True))
matrix_RFC=metrics.confusion_matrix(y_test, y_pred_rfc)
print('AUC',metrics.roc_auc_score(y_test, y_pred_rfc))
print('AUCROC:',metrics.roc_auc_score(y_test, y_pred_rfc))
print('AUC_ROC_CI',roc_auc_ci(y_test,y_pred_rfc))


tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_rfc).ravel()
print('tn:get_ipython().run_line_magic("d", " fp:%d fn:%d tp:%d' %(tn, fp, fn, tp))")
print('accuracy get_ipython().run_line_magic("f", " sensitivy %f, specificity %f, precision %f, f1score %f' \")
      get_ipython().run_line_magic("(((tp+tn)/(tp+tn+fn+fp)),(tp/(tp+fn)),tn/(tn+fp),tp/(tp+fp),", " 2*tp/(2*tp+fp+fn)))")


GradientBoostingClassifier().get_params()


n_estimators = [100, 200, 500]
max_features = ['auto', 'sqrt', 'log2','None']
max_depth = [3,4,5,6,7,8]

# define grid search
grid = dict(n_estimators = n_estimators,max_features=max_features,max_depth=max_depth)


start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, stratify = y, random_state=0)

scoring = {'AUC': 'roc_auc'}
gs = GridSearchCV(estimator = GradientBoostingClassifier(random_state=42),
                  param_grid=grid,
                  scoring=scoring, 
                  refit='AUC', 
                  return_train_score=True)
grid_result = gs.fit(X_train,y_train)
print('best params',grid_result.best_params_)
print('best AUC score:',grid_result.best_score_)
print('execution time:',time.time()-start_time)


model_gbc = GradientBoostingClassifier(max_depth=6,max_features='sqrt',n_estimators=500, random_state=42)
model_gbc.fit(X_train,y_train)
y_pred_gbc = model_gbc.predict(X_test)
print(metrics.classification_report(y_test,y_pred_gbc))
print(metrics.confusion_matrix(y_test,y_pred_gbc))
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), model_gbc.feature_importances_), X_train.columns),reverse=True))
matrix_GBC=metrics.confusion_matrix(y_test, y_pred_gbc)
print('AUCROC:',metrics.roc_auc_score(y_test, y_pred_gbc))
print('AUC_ROC_CI',roc_auc_ci(y_test,y_pred_gbc))


tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_gbc).ravel()
print('tn:get_ipython().run_line_magic("d", " fp:%d fn:%d tp:%d' %(tn, fp, fn, tp))")
print('accuracy get_ipython().run_line_magic("f", " sensitivy %f, specificity %f, precision %f, f1score %f' \")
      get_ipython().run_line_magic("(((tp+tn)/(tp+tn+fn+fp)),(tp/(tp+fn)),tn/(tn+fp),tp/(tp+fp),", " 2*tp/(2*tp+fp+fn)))")


X_test.columns


pd.DataFrame(list(zip(model_dtc.feature_importances_,X_test.columns)))


pd.DataFrame(list(zip(model_rfc.feature_importances_,X_test.columns)))


pd.DataFrame(list(zip(model_gbc.feature_importances_,X_test.columns)))


importance = pd.DataFrame(dict(features=X_test.columns,lr=model_lr.coef_[0],
                               dtc=model_dtc.feature_importances_,
                               rfc=model_rfc.feature_importances_,
                               gbc=model_gbc.feature_importances_)).set_index('features')
importance=importance.sort_values(by='gbc',ascending=False)


importance


#fig,ax = plt.subplots(2,1,figsize=(15,7))
#ax1,ax2=ax.flatten()
#ax1.set_xticks([])
#ax1.set_ylabel('Relative importance')
#ax2.set_xticks([])
#ax2.set_ylabel('Coeficients')
##sns.distplot()
#importance[['dtc','rfc','gbc']].plot(kind='bar',title='Features importance',ax=ax1,xticks=[]).xaxis.label.set_visible(False);
#importance[['lr']].plot(kind='bar',ax=ax2);
importance[['dtc','rfc','gbc']].plot(kind='bar',title='Variables importance')
plt.legend(['Decision Tree Classifier','Random Forest Classifier','Gradient Boosting Classifier'])
plt.ylabel('Relative importance')
plt.xlabel('Features');



plt.savefig(quality = 95, dpi = 'figure',fname = 'figure1.png')
plt.close()


#model_tree = DecisionTreeClassifier(class_weight={0: 4.0, 1: 0.571}, criterion='entropy',max_depth = 4, max_features='auto', random_state=42)
model_tree = DecisionTreeClassifier(criterion='entropy',max_depth = 4, max_features='auto', random_state=42)
model_tree.fit(X_test,y_test);


dot_data = StringIO()
export_graphviz(model_tree, out_file=dot_data, feature_names=X.columns, class_names=['0','1'],label='all',
                filled=False, rounded=True,
                special_characters=True,max_depth=6)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('trees/dtc1_arvore.png')
Image(graph.create_png())


y_test.value_counts()


model_tree = DecisionTreeClassifier(class_weight={0: 4.0, 1: 0.571}, criterion='entropy',max_depth = 3, max_features='auto', random_state=42)
model_tree.fit(X_test,y_test);


viz = dtreeviz(model_tree,X_test,y_test,target_name='dropout',feature_names=X.columns,class_names=['no','yes'])

viz.view()
viz.save('trees/dtc2_arvore.svg')


print(tree.export_text(model_dtc,decimals=1,max_depth=6,feature_names=['age', 'sex', 'dayswfreq', 'tbilled', 'maccess', 'nentries', 'cfreq','nrenewals', 'cref', 'months', 'rmonth']))


from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
tree.plot_tree(model_dtc,max_depth=3,feature_names=X_test.columns, class_names=['0','1'],filled=True);
fig.savefig('trees/dtree_dtc_arvore.png')


plt.rcParams['figure.figsize'] = [12, 15]


plot_partial_dependence(model_lr, X, X.columns,target=1);


plot_partial_dependence(model_dtc, X, X.columns,target=1);


plot_partial_dependence(model_rfc, X, X.columns,target=1);


plot_partial_dependence(model_gbc, X, X.columns,target=1);
