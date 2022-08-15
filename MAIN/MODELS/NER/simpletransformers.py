import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel,NERArgs

def create_model_args():
    args = NERArgs()
    args.num_train_epochs = 1
    args.learning_rate = 1e-4
    args.overwrite_output_dir =True
    args.train_batch_size = 10
    args.eval_batch_size = 10
    return args

def train_simpletransformers(data,sentence,word,pos,label,ner_model):
    
    ner_model_type = ner_model.split("/")[0]
    ner_model_name = ner_model.split("/")[1]

    print(f"Model type and model name is {ner_model_type,ner_model_type}")

    data[sentence] = LabelEncoder().fit_transform(data[sentence] )
    data[label] = data[label].str.upper()

    X= data[[sentence,word]]
    Y =data[label]
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)

    #building up train data and test data
    train_data = pd.DataFrame({"sentence_id":x_train[sentence],"words":x_train[word],"labels":y_train})
    test_data = pd.DataFrame({"sentence_id":x_test[sentence],"words":x_test[word],"labels":y_test})
    unique_label = data[label].unique().tolist()
    
    ## Training the model
    args = create_model_args()
    model = NERModel(ner_model_type, ner_model_name,labels=unique_label,args =args,use_cuda =False)
    model.train_model(train_data,eval_data = test_data,acc=accuracy_score)
    result, model_outputs, preds_list = model.eval_model(test_data)
    print(result)

    return result

