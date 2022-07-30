def start_prediction(path,usecase,ner_model):
    if usecase == "NER":
        if ner_model == "BILSTM":
            from MAIN.MODELS.NER.bilstm import predict
            out = predict(path,ner_model)
            return out
    else:
        pass