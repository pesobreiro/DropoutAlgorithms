import pandas as pd
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data

get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_excel('../dados/vinhos/vendas.xlsx')


# criar um id
df = df.reset_index()
df.head()


df.columns


df.columns = ['index', 'customerId', 'date', 'value','IVA', 'discountFinan', 'discountComer','valueWithIVA']


df.head()


#calcular a data maior
df.date.max()


df[df.customerId==7].head()


df[df.customerId==2].value.sum()


df.describe()


df.agg({'customerId':['count','mean','std','min','max']})


df.date.min()


df.date.max()


import datetime
df['year'] = df['date'].apply(lambda x: x.date().year)


df.head()


df.groupby(['year']).agg({'value':['sum']})


import numpy as np
import seaborn as sns
sns.barplot(x='year',y='value',data=df,estimator=np.sum)


summary = summary_data_from_transaction_data(df,customer_id_col='customerId',datetime_col='date',freq='M',
                                             monetary_value_col='value',observation_period_end='2019-09-11')
summary.head()


summary.describe()


summary.shape


summary.columns


summary['monetary_value'].astype(int).head()


from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
print(bgf)


bgf.summary


from lifetimes.plotting import plot_frequency_recency_matrix
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']= [10,10]

plot_frequency_recency_matrix(bgf,title="");


from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf,title="");


t=12
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], 
                                                        summary['T'])
summary.sort_values(by='predicted_purchases').tail(10)


from lifetimes.plotting import plot_period_transactions
plt.rcParams['figure.figsize']= [12,3]

plot_period_transactions(bgf);


from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(df, 'customerId', 'date',calibration_period_end='2019-03-31',
                                                   observation_period_end='2019-09-12' )


df.date.min()


df.date.max()


summary_cal_holdout.head()


from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout,title="");


summary.head(10)


summary.describe()


# Vamos prever o valor com o cliente número 11 está na posição 9
summary.iloc[9]


t = 12 #predict purchases in 10 periods
individual = summary.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
# 0.0576511


df.head()


from lifetimes.plotting import plot_history_alive

customerId = 3
days_since_birth = 24 #the number of time units since the birth we want to draw the p_alive
sp_trans = df.loc[df['customerId'] == customerId]
plot_history_alive(bgf, days_since_birth, sp_trans, 'date',freq='M');


df.columns


df.loc[df.customerId == 3].agg({'valueWithIVA':['sum','count','mean','std','min','max']})


summary.head(10)


summary_ggf = summary.loc[(summary.frequency > 0) & (summary.monetary_value > 0)]


summary_ggf.columns


summary_ggf[['frequency','monetary_value']].corr()


summary_ggf.monetary_value.hist()


from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(summary_ggf['frequency'],summary_ggf['monetary_value'])


ggf.conditional_expected_average_profit(summary_ggf['frequency'],summary_ggf['monetary_value']).head(10)


bgf.fit(summary_ggf['frequency'], summary_ggf['recency'], summary_ggf['T'])


bgf.fit(summary_ggf['frequency'], summary_ggf['recency'], summary_ggf['T'])

ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_ggf['frequency'],
    summary_ggf['recency'],
    summary_ggf['T'],
    summary_ggf['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10)


ggf_CLV = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_ggf['frequency'],
    summary_ggf['recency'],
    summary_ggf['T'],
    summary_ggf['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
)


ggf_CLV.head(10)


# transformar Series em dataframe 
ggf_CLV=pd.DataFrame(ggf_CLV)
#ggf=pd.DataFrame(ggf)


dadosTodos=pd.merge(summary,ggf_CLV, left_index=True, right_index=True)


dadosTodos.shape


dadosTodos.head().reset_index()


X = dadosTodos.copy()


X=X.reset_index()


X


vars=['frequency','recency','monetary_value','clv']
#vars=['monetary_value','clv']


unscaled = X.copy()


from sklearn import preprocessing
from scipy.spatial.distance import cdist
preprocessing.Normalizer()
#Standardizar os valores
scaled = preprocessing.Normalizer().fit_transform(X[vars]) #devolve numpy array
X[['s_frequency','s_recency','s_monetary_value','s_clv']]=pd.DataFrame(scaled, columns=X[vars].columns) #converter novamente para um dataframe


X.head()


varsNorm=['s_frequency','s_recency','s_monetary_value','s_clv']


from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import numpy as np
plt.rcParams['figure.figsize']= [12,3]
#Finding optimal no. of clusters
clusters=range(1,20)
meanDistortions=[]
 
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(X[varsNorm])
    prediction=model.predict(X[varsNorm])
    meanDistortions.append(sum(np.min(cdist(X[varsNorm], model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
#plt.cla()
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
#plt.title('Selecting k with the Elbow Method')


cluster = KMeans(n_clusters=3)

cluster.fit(X[varsNorm])
X['clusterKmeans']=cluster.predict(X[varsNorm])
print(X.clusterKmeans.value_counts())


from sklearn.cluster import AgglomerativeClustering 
clusterWard = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
clusterWard.fit(X[varsNorm])
X['clusterWard']=clusterWard.fit_predict(X[varsNorm])
print(X.clusterWard.value_counts())


X.columns


from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = [13, 4]


X[varsNorm].iloc[:,0:2].head(1)


X[varsNorm].iloc[:,2:4].head(1)


pca=PCA(n_components=2)
pcaComponents = pca.fit_transform(X[varsNorm])
principalDf = pd.DataFrame(data = pcaComponents, columns = ['principal component 1', 'principal component 2'])


X[['X_pca','Y_pca']]=pca.fit_transform(X[varsNorm])


pca=PCA(n_components=2)
X[['X_pca','Y_pca']]=pca.fit_transform(X[varsNorm])

fig, ax = plt.subplots()
for cluster in X.clusterKmeans.unique():
    x = X['X_pca'].loc[X.clusterKmeans == cluster]
    y = X['Y_pca'].loc[X.clusterKmeans == cluster]
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')

'''
fig, ax = plt.subplots()
for cluster in X.clusterKmeans.unique():
    x = X['monetary_value'].loc[X.clusterKmeans == cluster]
    y = X['clv'].loc[X.clusterKmeans == cluster]
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')
'''

ax.legend()
ax.grid(True)
ax.set_title('Clusters clientes K-Means')
plt.show()


X.head()


fig, ax = plt.subplots()
for cluster in X.clusterWard.unique():
    x = X['X_pca'].loc[X.clusterWard == cluster]
    y = X['Y_pca'].loc[X.clusterWard == cluster]
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')

L=ax.legend()
ax.grid(True)

L.get_texts()[2].set_text('cluster 3')
L.get_texts()[1].set_text('cluster 1')
L.get_texts()[0].set_text('cluster 2')
# ax.set_title('Clusters clientes Ward')
plt.show()


fig, ax = plt.subplots()
for cluster in X.clusterWard.unique():
    x = X['monetary_value'].loc[X.clusterWard == cluster]
    y = X['clv'].loc[X.clusterWard == cluster]
    ax.scatter(x, y, label=cluster, alpha=1, edgecolors='none')

L=ax.legend()
ax.grid(True)

L.get_texts()[2].set_text('cluster 3')
L.get_texts()[1].set_text('cluster 1')
L.get_texts()[0].set_text('cluster 2')
#ax.set_title('Clusters clientes Ward')
plt.show()


import scipy.cluster.hierarchy as sch
#Lets create a dendrogram variable linkage is actually the algorithm 
#itself of hierarchical clustering and then in linkage we have to #specify on which data we apply and engage. This is X dataset
plt.rcParams['figure.figsize'] = [12, 3]

dendrogram = sch.dendrogram(sch.linkage(X[varsNorm], method  = "ward"))
#plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


X


X.groupby(['clusterWard']).agg({'frequency':['count','mean','std'],
                            'recency':['mean','std'],
                            'T':['mean','std'],
                            'monetary_value':['mean','std'],
                            'predicted_purchases':['mean','std'],
                            'clv':['mean','std']})


X.groupby(['clusterKmeans']).agg({'frequency':['count','mean','std'],
                            'recency':['mean','std'],
                            'T':['mean','std'],
                            'monetary_value':['mean','std'],
                            'predicted_purchases':['mean','std'],
                            'clv':['mean','std']})


X.columns


Y = X.copy()


cols = ['clv']


X.columns


X.r_quartile.value_counts()


X['r_quartile'] = pd.qcut(X['recency'], q=[0,.2,.4,.6,.8,1], labels=['5','4','3','2','1'])
X['f_quartile'] = pd.qcut(X['frequency'], q=[0,.2,.4,.6,.8,1], labels=['1','2','3','4'],duplicates='drop')
X['m_quartile'] = pd.qcut(X['monetary_value'], q=[0,.2,.4,.6,.8,1], labels=['1','2','3','4','5'])


X['RFM_Score'] = X.r_quartile.astype(str)+ X.f_quartile.astype(str) + X.m_quartile.astype(str)
X.head()


X[X.RFM_Score=='555'].sort_values('monetary_value',ascending=False).head()


X.RFM_Score.value_counts().count()


X.columns


pd.crosstab(X.m_quartile,X.r_quartile)


X.pivot_table(index='r_quartile', columns=['f_quartile','m_quartile'],aggfunc={'r_quartile':len}, fill_value=0)


X.groupby(['clusterKmeans']).agg({'frequency':['count','mean','std'],
                            'recency':['mean','std'],
                            'T':['mean','std'],
                            'monetary_value':['mean','std'],
                            'predicted_purchases':['mean','std'],
                            'clv':['mean','std']})


X.agg({'r_quartile':['count','mean','std','min','max'],
        'f_quartile':['count','mean','std'],
        'm_quartile':['count','mean','std']})
