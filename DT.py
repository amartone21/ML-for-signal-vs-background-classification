import ROOT
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = (14, 7)
import root_pandas
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from My_config import fetch_configuration 
from funzioni import compare_train_test

#Under100
conf1= '100'
png1 = './'+ conf1 

#Under250
conf2= 'U250'
png2 = './'+ conf2

#Over250
conf3= 'O250'
png3 = './'+ conf3


config=conf3
png_name= png3
config_dict=fetch_configuration()
truth1='Higgs_truth1'  
truth2='Higgs_truth2'

#importo il dataset
data = root_pandas.read_root('./candidati1200_3.root', 'toFormat')
data_new = root_pandas.read_root('./candidati1200_3.root', 'toFormat')
data_pretrain = data_new

#seleziono solo gli eventi con le features compatibili con la fisica del problema
data= data.query('jet1_mass>=80' and 'jet1_mass<= 160')
data=data.query( 'jet2_mass>=80' and 'jet2_mass<= 160')
data = data.query(config_dict[config]['bin'])

#il secondo dataset prende tutte le variabili
data_new = data_new[config_dict[config]['variables']]

#splitting 
X_train, X_test = train_test_split(data, test_size=0.3)

#Prendo X_train e X_test con tutte le colonne 
X_train_all_variables, X_test_all_variables = X_train.query(config_dict[config]['presel']), X_test.query(config_dict[config]['presel'])
X_train, X_test = X_train_all_variables[config_dict[config]['variables']], X_test_all_variables[config_dict[config]['variables']]

#creo le etichette di verita
y_train, y_test = X_train_all_variables[truth1].values, X_test_all_variables[truth1].values

#questo e' un test
if(np.count_nonzero(y_train) == 0 | np.count_nonzero(y_test) == 0):
   print("No data in configuration: ", config)


 #Hyperparameters:
n_estimators = 130
learning_rate = 0.1
max_depth = 4
min_child_weight = 4
reg_alpha = .01

# Early stopping
early_stopping_rounds = 15

# Define model
model_bdt = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                              max_depth=max_depth, min_child_weight=min_child_weight, 
                              reg_alpha=reg_alpha)

#Last in list is used for early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]

# Fit with early stopping
model_bdt.fit(X_train, y_train, eval_metric=["logloss"], eval_set=eval_set, 
              early_stopping_rounds=early_stopping_rounds, verbose=False)

#Save Model
model_bdt.save_model('BDT_' + config +  '.model')

y_pred = model_bdt.predict_proba(X_test)[:, 1]
y_pred_data = model_bdt.predict_proba(data_new)[:, 1]
bkpred = model_bdt.predict_proba(data_new)[:, 0]

# evaluate predictions
tpr10 =[]
tpr1 =[]
t10=[]
t1=[]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
for idx in range(len(tpr)):
   if (fpr[idx])<0.1:        
      tpr10.append(tpr[idx])
      t10.append(thresholds[idx])
   if (fpr[idx])<0.01:        
      tpr1.append(tpr[idx])
      t1.append(thresholds[idx])

#salvo i tpr
np.savetxt('tpr10.csv', tpr10, delimiter=',')
np.savetxt('tpr1.csv', tpr1, delimiter=',')

#salvo le soglie
np.savetxt('tpr_soglie10.csv', t10, delimiter=',')
np.savetxt('tpr_soglie1.csv', t1, delimiter=',')

#salvo i fpr
np.savetxt('fpr10.csv', fpr, delimiter=',')

#creo file root
data_new['Signal_Score_'+ config]=y_pred_data
data_new['Bkg_score_'+ config]= bkpred
data_new["Higgs_resolved"]=data_pretrain['Higgs_resolved']
data_new["Higgs_merged"]=data_pretrain['Higgs_merged']
data_new["Higgs_truth1"]=data_pretrain['Higgs_truth1']
data_new.to_root('dataset_m1200_with_score'  + config+  '.root','ml' )

#plot logloss
res = model_bdt.evals_result()
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(range(0, len(res['validation_0']['logloss'])), res['validation_0']['logloss'], label='Train')
ax.plot(range(0, len(res['validation_1']['logloss'])), res['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('LogLoss')
plt.title(' Andamento LogLoss'+  config)
fig.savefig( png_name + 'logloss'+ '.png')
  
#funzione di visualizazione output della bdt   
compare_train_test(model_bdt, X_train, y_train, X_test, y_test, config, png_name)

#feature importance plot
importance_plot = xgb.plot_importance(model_bdt)
importance_plot.figure.savefig(png_name)

#roc curve
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
ax_roc.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax_roc.set_xlim([0, 1.0])
ax_roc.set_ylim([0, 1.0])
ax_roc.set_xlabel('false positive rate')
ax_roc.set_ylabel('true positive rate')
ax_roc.set_title('receiver operating curve'+ config  )
ax_roc.legend(loc="lower right")
fig_roc.savefig(  'roc'+ config  + '.png', format='png')
   





