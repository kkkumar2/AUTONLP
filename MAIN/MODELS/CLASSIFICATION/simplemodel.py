from MAIN.MODELS.CLASSIFICATION.utils import load_data_mc

import lightgbm as lgbm   
from sklearn.svm import SVC
import numpy as np


def mcrun(csv,features,labels,model):
    if model.lower() in 'lightgbm':
        config = load_data_mc(csv,features,labels,model)
        # value = {'Train':train,'Valid':valid,"parameters":parameters,"unique_labels":unique_labels_name}
        classifier = lgbm.train(config['parameters'],
                                config['Train'],
                                valid_sets = config['Valid'],
                                num_boost_round=50,
                                early_stopping_rounds=3)
        val_pred = classifier.predict(config['Valid'])
        if len(config['unique_labels']) > 2:
            y_pred = [1 if pred>0.5 else 0 for pred in val_pred]

        else:
            y_pred = [np.argmax(pred) for pred in val_pred]
    
    if model.lower() in 'svc':pass