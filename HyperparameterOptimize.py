from modelFactory import HyperparameterTune
from metrics import dice_coef_loss
import DataLoad 
from sklearn.model_selection import train_test_split
import talos

X,Y = DataLoad.load3D_XY()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


p = {'optimizer': ['Nadam', 'Adam'], #not used yet
     'losses': ['logcosh', dice_coef_loss],
     'epochs': [10, 20],
     'lr':[1e-2, 1e-3, 1e-4],
     'lr_decay': [0.1, 0.25, 0.5]}


scan_object = talos.Scan(X_train, y_train, model=HyperparameterTune, params=p, fraction_limit=0.1)