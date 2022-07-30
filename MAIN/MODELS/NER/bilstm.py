def BILSTM_MODEL(max_len,n_words,max_len_char,n_chars,n_tags):
    from keras.models import Model, Input
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
    from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D

        # input and embedding for words
    word_in = Input(shape=(max_len,))
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                        input_length=max_len, mask_zero=True)(word_in)

    # input and embeddings for characters
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                            input_length=max_len_char, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                    recurrent_dropout=0.5))(emb_char)

    # main LSTM
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

    inputs=[word_in,char_in]
    model = Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model

def preprocess(data,sentence,word,pos,label):
    from keras_preprocessing.sequence import pad_sequences
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    words = list(set(data[word].values))
    n_words = len(words)
    tags = list(set(data[label].values))
    n_tags = len(tags)

    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s[word].values.tolist(),
                                                           s[pos].values.tolist(),
                                                           s[label].values.tolist())]
                
    grouped = data.groupby(sentence).apply(agg_func)
    sentences = [s for s in grouped]

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}    
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    # max_len = [len(s) for s in sentence]
    # max_len = max(max_len)
    max_len = 75
    X_word = [[word2idx[w[0]] for w in s] for s in sentences]
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')
    
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)

    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    max_len_char = 10
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

    X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
    X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)

    print("Building Bilstm model")
    model = BILSTM_MODEL(max_len,n_words,max_len_char,n_chars,n_tags)

    print(X_word_tr)

    print("Training started")
    history = model.fit([X_word_tr,
                     np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
                    batch_size=32, epochs=1, validation_split=0.1, verbose=1)

    hist = pd.DataFrame(history.history)

    print(f"Maximum validation accuracy is {hist['val_acc'].max()}")

    return hist['val_acc'].max()


def predict():
    pass
