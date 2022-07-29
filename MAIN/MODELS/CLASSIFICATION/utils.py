import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgbm

class Config:
    Batch_Size = 42
    Vocab_Size = 10_000
    Embadding_Dim = 64
    # Machine learning
    # ----- lightgbm
    Objective = ['binary','multiclass']
    Metric = 'auc'
    Boosting = 'drat'
    Num_Leaves = 31
    Feature_Fraction = 0.5
    Bagging_Fraction = 0.5
    Bagging_Freq = 20
    Learning_Rate = 0.05
    Verbose = 0 



def load_data_tf(csv_file,features,labels):
    df = pd.read_csv(csv_file)
    features = df[features]
    labels = df[labels]

    unique_labels_name = labels.unique().tolist() # labels names in DataFrame
    labels = labels.map({name:i for i,name in enumerate(unique_labels_name)}) # mapping string name to int ex possitve replace to 1
    x_train,y_train,x_test,y_test = train_test_split((features,labels),test_size=0.2,random_state=38)

    train = tf.data.Dataset.from_tensor_slices((x_train.values,y_train.values)) # load the data tensorflow
    test = tf.data.Dataset.from_tensor_slices((x_test.values,y_test.values))

    train = train.shuffle(len(x_train)).batch(Config.Batch_Size).prefetch(tf.data.AUTOTUNE) # shuffle and take batch 
    train = train.batch(Config.Batch_Size).prefetch(tf.data.AUTOTUNE)

    encoder = tf.keras.layers.TextVectorization(max_tokens=Config.Vocab_Size) # vectorization with automate added <pad> and <unk>
    encoder.adapt(train.map(lambda text,labels: text)) # Vectorizatin  only apply features text 

    value = {"Train":train,"Test":test,"encoder": encoder,"output_dim":len(unique_labels_name),"embadding_dim":Config.Embadding_Dim}

    return value

def load_data_mc(csv_file,feature,labels,model):
    df = pd.read_csv(csv_file)
    features = df[features]
    labels = df[labels]

    unique_labels_name = labels.unique().tolist()
    labels = labels.map({name:i for i,name in enumerate(unique_labels_name)})
    x_train,y_train,x_test,y_test = train_test_split((features,labels),test_size=0.2,random_state=38)

    tfidf = TfidfVectorizer(strip_accents=None,
                                lowercase=False,
                                preprocessor=None)
    tfidf.fit(x_train)
    x_train = tfidf.transform(x_train)
    x_test = tfidf.transform(x_test)

    


    if model.lower() in 'lightgbm':
        if len(unique_labels_name) > 2:
            objective = Config.Objective[0]
        else:
            objective = Config.Objective[1]

        parameters = {
            'objective': objective,
            'metric': Config.Metric,
            'boosting': Config.Boosting,
            'num_leaves': Config.Num_Leaves,
            'feature_fraction': Config.Feature_Fraction,
            'bagging_fraction': Config.Bagging_Fraction,
            'bagging_freq': Config.Bagging_Freq,
            'learning_rate': Config.Learning_Rate,
            'verbose': Config.Verbose
                        }
        
        train = lgbm.Dataset(x_train,y_train)
        valid = lgbm.Dataset(x_test,y_test)

        value = {'Train':train,'Valid':valid,"parameters":parameters,"unique_labels":unique_labels_name}
        return value

    else:
        return {"Trian":train,"Valid":valid,"unique_labels":unique_labels_name}

                


