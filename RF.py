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
from sklearn import tree
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from config import fetch_configuration 
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#Here I define some configuration variables, in order to not write them all the time

#Under100
conf1= '100'


#Under250
conf2= 'U250'


#Over250
conf3= 'O250'

#Here I define the desired configuration to work with, the truth lable and the path to the file 
config=conf3
path= './130/'
png_name= path + config
config_dict=fetch_configuration()
truth1='Higgs_truth1'  


#import dataset
data = root_pandas.read_root('./candidati1200_3.root', 'toFormat')
data_new = root_pandas.read_root('./candidati1200_3.root', 'toFormat')
data_pretrain = data_new

#Event selection based on the physics
data= data.query('jet1_mass>=80' and 'jet1_mass<= 160')
data=data.query( 'jet2_mass>=80' and 'jet2_mass<= 160')
data = data.query(config_dict[config]['bin'])

#Second dataset takes all varibles 
data_new = data_new[config_dict[config]['variables']]

#splitting 
X_train, X_test = train_test_split(data, test_size=0.3)

#Take X_train e X_test with every column
X_train_all_variables, X_test_all_variables = X_train.query(config_dict[config]['presel']), X_test.query(config_dict[config]['presel'])
X_train, X_test = X_train_all_variables[config_dict[config]['variables']], X_test_all_variables[config_dict[config]['variables']]

#Define truth lables
y_train, y_test = X_train_all_variables[truth1].values, X_test_all_variables[truth1].values


#Just a test to makesure there is data
if(np.count_nonzero(y_train) == 0 | np.count_nonzero(y_test) == 0):
   print("No data in configuration: ", config)


 #Hyperparameters:
n_estimators = 200
learning_rate = 0.1
max_depth = 4
min_child_weight = 4
reg_alpha = .01

# Early stopping
early_stopping_rounds = 15


# Define model
model_rf = RandomForestClassifier(n_estimators= n_estimators, 
                                  criterion='entropy')

#Last in list is used for early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]

# Fit with early stopping
model_rf.fit(X_train, y_train)


y_pred = model_rf.predict_proba(X_test)[:, 1]
y_pred_data = model_rf.predict_proba(data_new)[:, 1]
bkpred = model_rf.predict_proba(data_new)[:, 0]

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

#save tpr
np.savetxt(path+ 'tpr10_' + config +'.csv', tpr10, delimiter=',')
np.savetxt(path+'tpr1_'+config +'.csv', tpr1, delimiter=',')

#save threhsolds
np.savetxt(path+'tpr_soglie10_'+config+'.csv', t10, delimiter=',')
np.savetxt(path+'tpr_soglie1_'+config+'.csv', t1, delimiter=',')

#save fpr
np.savetxt(path+'fpr10_'+config+'.csv', fpr, delimiter=',')



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
fig_roc.savefig( path  + config + '.png', format='png')






