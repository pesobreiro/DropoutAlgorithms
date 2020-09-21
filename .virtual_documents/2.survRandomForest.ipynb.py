from IPython.display import HTML
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
import seaborn as sns

df = pd.read_excel('../dados/dadosSociosTratados.xlsx',index_col=0)


df.sexo.value_counts()


df.describe()


df.info()


df['ultimoPagamento'] = pd.to_datetime(df['ultimoPagamento'],format='get_ipython().run_line_magic("Y-%m-%d", " %H:%M', errors='coerce')")


df['anoUltimoPagamento']=df['ultimoPagamento'].apply(lambda x: x.year)


df.anoUltimoPagamento.unique()


df.anoUltimoPagamento=df.anoUltimoPagamento.fillna(0)


df['anoUltimoPagamento']=df.anoUltimoPagamento.astype(int)


df.drop(columns=['nome','dataAdesao','ultimoPagamento','contribuinte','dataNascimento','mes','profissao',
                 'codPostal','categoria','ultimaQuota','diasUltimoPagamento'],inplace=True)


df.info()


df.estadoCivil.value_counts()


dfCurvas = df.copy()
df = pd.get_dummies(df, columns=['sexo','estadoCivil','escaloesTotalJogos'],drop_first=True)


# Creating the time and event columns
time_column = 'anosSocio'
event_column = 'abandonou'

# Extracting the features
features = np.setdiff1d(df.columns, [time_column, event_column] ).tolist()


# Checking for null values
N_null = sum(df[features].isnull().sum())
print("The raw_dataset contains {} null values".format(N_null)) #0 null values


# Removing duplicates if there exist
N_dupli = sum(df.duplicated(keep='first'))
df = df.drop_duplicates(keep='first').reset_index(drop=True)
print("The raw_dataset contains {} duplicates".format(N_dupli))

# Number of samples in the dataset
N = df.shape[0]


df.columns


from pysurvival.utils.display import correlation_matrix
correlation_matrix(df[features], figure_size=(10,10), text_fontsize=8)


to_remove = ['totalJogos', 'idaEstadio']
features = np.setdiff1d(features, to_remove).tolist()


# Building training and testing sets
from sklearn.model_selection import train_test_split
index_train, index_test = train_test_split( range(N), test_size = 0.4)
data_train = df.loc[index_train].reset_index( drop = True )
data_test  = df.loc[index_test].reset_index( drop = True )

# Creating the X, T and E inputs
X_train, X_test = df[features], data_test[features]
T_train, T_test = df[time_column], data_test[time_column]
E_train, E_test = df[event_column], data_test[event_column]


#from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
# Fitting the model
csf = RandomSurvivalForestModel(num_trees=200)
csf.fit(X_train, T_train, E_train, max_features='sqrt',
        max_depth=5, min_node_size=20)


csf.variable_importance_table


from pysurvival.utils.metrics import concordance_index
c_index = concordance_index(csf, X_test, T_test, E_test)
print('C-index: {:.2f}'.format(c_index)) #0.83


from pysurvival.utils.display import integrated_brier_score
ibs = integrated_brier_score(csf, X_test, T_test, E_test, t_max=12,
    figure_size=(12,5))
print('IBS: {:.2f}'.format(ibs))


to_remove = ['estadoCivil_outro', 'ano']
features = np.setdiff1d(features, to_remove).tolist()


# Creating the X, T and E inputs
X_train, X_test = df[features], data_test[features]
T_train, T_test = df[time_column], data_test[time_column]
E_train, E_test = df[event_column], data_test[event_column]


csf = RandomSurvivalForestModel(num_trees=200)
csf.fit(X_train, T_train, E_train, max_features='sqrt',
        max_depth=5, min_node_size=20)


csf.variable_importance_table


from pysurvival.utils.metrics import concordance_index
c_index = concordance_index(csf, X_test, T_test, E_test)
print('C-index: {:.2f}'.format(c_index)) #0.83


from pysurvival.utils.display import integrated_brier_score
ibs = integrated_brier_score(csf, X_test, T_test, E_test, t_max=12,
    figure_size=(12,5))
print('IBS: {:.2f}'.format(ibs))


to_remove = ['idade']
features = np.setdiff1d(features, to_remove).tolist()


# Creating the X, T and E inputs
X_train, X_test = df[features], data_test[features]
T_train, T_test = df[time_column], data_test[time_column]
E_train, E_test = df[event_column], data_test[event_column]


csf = RandomSurvivalForestModel(num_trees=200)
csf.fit(X_train, T_train, E_train, max_features='sqrt',
        max_depth=5, min_node_size=20)


csf.variable_importance_table


from pysurvival.utils.metrics import concordance_index
c_index = concordance_index(csf, X_test, T_test, E_test)
print('C-index: {:.2f}'.format(c_index)) #0.83


from pysurvival.utils.display import integrated_brier_score
ibs = integrated_brier_score(csf, X_test, T_test, E_test, t_max=12,
    figure_size=(12,5))
print('IBS: {:.2f}'.format(ibs))


from pysurvival.utils.display import compare_to_actual
results = compare_to_actual(csf, X_test, T_test, E_test, is_at_risk = False,  figure_size=(12, 5), metrics = ['rmse', 'mean', 'median'])


from pysurvival.utils.display import create_risk_groups

risk_groups = create_risk_groups(model=csf, X=X_test,
    use_log = False, num_bins=30, figure_size=(20, 4))


def curvaSobrevivencia(dados,coluna):
    ax = plt.subplot(111)
    plt.rcParams['figure.figsize'] = [12, 5]
    for item in dados[coluna].unique():
        ix = dados[coluna] == item
        kmf.fit(T.loc[ix], C.loc[ix], label=str(item))
        ax = kmf.plot(ax=ax)


from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.statistics import pairwise_logrank_test

kmf = KaplanMeierFitter()
T = dfCurvas['anosSocio']
C = dfCurvas['abandonou']
kmf.fit(T,C,label="Abandono dos sócios");


tabela=pd.concat([kmf.event_table.reset_index(), 
           kmf.conditional_time_to_event_.reset_index(),
           kmf.survival_function_.reset_index()],axis=1)


tabela.columns = ['event_at', 'removed', 'observed', 'censored', 'entrance', 'at_risk','timeline',
                  'median duration remaining to event','timeline', 'Abandono dos sócios']


tabela.head(12)


plt.rcParams['figure.figsize'] = [12, 5]

kmf.plot();


print(dfCurvas.sexo.value_counts())
curvaSobrevivencia(dfCurvas,'sexo')


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas.sexo,event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas.sexo,event_observed=C)
results.print_summary()


dfCurvas.mesesUP.describe()


var='mesesUP'
varEscalao='escMesesUP'
dfCurvas[varEscalao]=''
for index, cliente in dfCurvas.iterrows():
    #se a variável tiver o valor 1 colocar na nova variável a descrição da atividade
    if cliente[var] <= 2: 
        dfCurvas.at[index,varEscalao]=var+' less than 2'
    elif (cliente[var] > 2) & (cliente[var] <= 4):
        dfCurvas.at[index,varEscalao]=var+' greather than 2 and less 4'
    elif (cliente[var] > 4) & (cliente[var] <= 17):
        dfCurvas.at[index,varEscalao]=var + ' greather than 4 and less 17'
    elif (cliente[var] > 17):
        dfCurvas.at[index,varEscalao]=var + ' greather than 17'


dfCurvas.escMesesUP.value_counts()


curvaSobrevivencia(dfCurvas,varEscalao)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


dfCurvas.valorTotal.describe()


var='valorTotal'
varEscalao='escValorTotal'
dfCurvas[varEscalao]=''
for index, cliente in dfCurvas.iterrows():
    #se a variável tiver o valor 1 colocar na nova variável a descrição da atividade
    if cliente[var] <= 5: 
        dfCurvas.at[index,varEscalao]=var+' less than 5'
    elif (cliente[var] > 5) & (cliente[var] <= 53):
        dfCurvas.at[index,varEscalao]=var+' greather than 5 and less 53'
    elif (cliente[var] > 53) & (cliente[var] <= 448):
        dfCurvas.at[index,varEscalao]=var + ' greather than 53 and less 448'
    elif (cliente[var] > 448):
        dfCurvas.at[index,varEscalao]=var + ' greather than 448'


dfCurvas[varEscalao].value_counts()


curvaSobrevivencia(dfCurvas,varEscalao)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


varEscalao='quotaMensal'
dfCurvas[varEscalao].describe()


dfCurvas[varEscalao].value_counts()


curvaSobrevivencia(dfCurvas,varEscalao)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


var='jogosEpoca'
varEscalao='escJogosEpoca'
dfCurvas[var].describe()


dfCurvas[varEscalao]=''
for index, cliente in dfCurvas.iterrows():
    #se a variável tiver o valor 1 colocar na nova variável a descrição da atividade
    if cliente[var] <= 2: 
        dfCurvas.at[index,varEscalao]=var+' less than 2'
    elif (cliente[var] > 2):
        dfCurvas.at[index,varEscalao]=var + ' greather than 2'


dfCurvas[varEscalao].value_counts()


curvaSobrevivencia(dfCurvas,varEscalao)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[varEscalao],event_observed=C)
results.print_summary()


var='escaloesTotalJogos'
dfCurvas[var].describe()


dfCurvas[var].value_counts()


curvaSobrevivencia(dfCurvas,var)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[var],event_observed=C)
results.print_summary()


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[var],event_observed=C)
results.print_summary()


var='estadoCivil'
dfCurvas[var].describe()


dfCurvas[var].value_counts()


curvaSobrevivencia(dfCurvas,var)


results=multivariate_logrank_test(event_durations=T,groups=dfCurvas[var],event_observed=C)
results.print_summary


results=pairwise_logrank_test(event_durations=T,groups=dfCurvas[var],event_observed=C)
results.print_summary()
