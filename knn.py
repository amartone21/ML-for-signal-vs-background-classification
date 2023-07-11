import ROOT
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = (14, 7)
import root_pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
from config import fetch_configuration
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline

#importo dal dizionario le variabili che mi servono
config_dict=fetch_configuration()
config2 ='dijet_mass_pt'
config3 = 'btag_pt'
config= 'all'
#scelgo se analizzare ak4 o ak8
truth1='Higgs_truth1'  


#importo il dataset
data = root_pandas.read_root('./candidati1200_3.root', 'toFormat')

#gestisco dimensioni del dataset
''' mantengo le proporzioni di segnale e fondo originali (1 a 56) ma riduco le dimensioni del dataset a 14782
 in modo da ottimizzare k come radice del numero di eventi di train '''
signal_df_new = data.query("Higgs_truth1==1").sample(n= 263 )
fondo_df_new = (data.query("Higgs_truth1 == 0")).sample(n= 56*len(signal_df_new))
df= [fondo_df_new, signal_df_new]
df_new = pd.concat(df)

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

#hyperparametri
n_neighbors = 102
weights = 'uniform'
weights2 = 'distance'
p = 1
metric= 'minkowski' 


#normalizzo i dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#alleno il classificatore
classifier = KNeighborsClassifier(n_neighbors = n_neighbors,
                                  weights= weights)


# riduci dimensioni con  PCA
pca = make_pipeline(StandardScaler(),
                    PCA(n_components=4))

pca.fit (X_train, y_train)
X_train_prime =  pca.transform (X_train)
X_test_prime = pca.transform ( X_test)



#fit del clasificatore
classifier.fit(X_train, y_train)

# predizioni sul Test set 
y_pred = classifier.predict_proba(X_test)[:,1]




#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
ax_roc.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax_roc.set_xlim([0, 1.0])
ax_roc.set_ylim([0, 1.0])
ax_roc.set_xlabel('false positive rate')
ax_roc.set_ylabel('true positive rate')
ax_roc.set_title('receiver operating curve' )
ax_roc.legend(loc="lower right")
fig_roc.savefig(  'roc.png', format='png')

# evaluate predictions
tpr10 =[]
tpr1 =[]
t10=[]
t1=[]

for idx in range(len(tpr)):
   if (fpr[idx])<0.1:
      tpr10.append(tpr[idx])
      t10.append(thresholds[idx])
   if (fpr[idx])<0.01:
      tpr1.append(tpr[idx])
      t1.append(thresholds[idx])

print(' il max al 10 in configurazione', np.amax(tpr10))
print(' il max al 1:', np.amax(tpr1))
