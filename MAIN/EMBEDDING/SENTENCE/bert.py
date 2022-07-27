def bert_similarity(data,mode='cbow'):
    
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('msmacro-distilroberta-base-v2')
    
    train_data = list(map(str.lower,data))
    vectors = model.encode(train_data,convert_to_tensor=False)
    
    return vectors