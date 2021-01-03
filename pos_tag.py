import numpy as np
import keras
from nltk.corpus import treebank
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


from os import walk


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DATA UTILITIES
"""
def gen_feat(sentence_terms, index):
    term = sentence_terms[index]
    return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'is_capitalized': term[0].upper() == term[0],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'prefix-3': term[:3],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'suffix-3': term[-3:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }

def untag(tagged_sentence):
    return [w for w, _ in tagged_sentence]

def str2dct(tagged_sentences):
    x, y = [], []
    for pos_tags in tagged_sentences:
            for index, (term, class_) in enumerate(pos_tags):
                x.append(gen_feat(untag(pos_tags), index))
                y.append(class_)
    return x, y

def str2dct2(sentence):
    x = []
    for index, word in enumerate(sentence):
        x.append(gen_feat(sentence, index))
    return x


def dct2arr(xtrn, xtst, xval):
    dict_encoder = DictVectorizer(sparse=False)
    dict_encoder.fit(xtrn + xtst + xval)
    xtrn = dict_encoder.transform(xtrn)
    xtst = dict_encoder.transform(xtst)
    xval = dict_encoder.transform(xval)
    return dict_encoder, xtrn, xtst, xval

def catenc(ytrn, ytst, yval):
    label_encoder = LabelEncoder()
    label_encoder.fit(ytrn + ytst + yval)
    ytrn = label_encoder.transform(ytrn)
    ytst = label_encoder.transform(ytst)
    yval = label_encoder.transform(yval)

    return label_encoder, ytrn, ytst, yval

def ohenc(ytrn, ytst, yval):
    ytrn = np_utils.to_categorical(ytrn)
    ytst = np_utils.to_categorical(ytst)
    yval = np_utils.to_categorical(yval)
    return ytrn, ytst, yval

def ttvsplit(stc, trn, tst, val):
    ntrn = int(trn * len(stc))
    ntst = int(tst * len(stc))
    trnstc = stc[:ntrn]
    tststc = stc[ntrn:ntrn+ntst]
    valstc = stc[ntst:]
    return trnstc, tststc, valstc


def parsebrown():
    fpath = "brown-universal.txt"  
    data = open(fpath, "r")
    txt = data.read();
    data.close()
    stc = [s.split("\n") for s in txt.split("\n\n")]
    out = []
    for s in stc:
        tmp = [x.split('\t') for x in stc[1][1:]]
        out.append(tmp)
    return out
  



"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODEL
"""

def build_model(input_dim, hidden_neurons, output_dim):
    model = Sequential([
        Dense(1024, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():

    # """
    # ++++++++++++++++++++++++++++++++++++++++++
    # DATA PREPROCESSING
    # """

    #########
    # EITHER
    sentences = treebank.tagged_sents()

    # OR
    # sentences = parsebrown() # have to dl brown corpus ("brown-universal.txt") and change path in parsebrown function
    #########




    # trnstc, tststc, valstc  = ttvsplit(sentences[0:50000], .6, .3, .1)
    trnstc, tststc, valstc  = ttvsplit(sentences, .6, .3, .1)

    xtrn, ytrn          = str2dct(trnstc)
    xtst, ytst          = str2dct(tststc)
    xval, yval          = str2dct(valstc)

    dict_encoder, xtrn, xtst, xval     = dct2arr(xtrn, xtst, xval)

    label_encoder, ytrn, ytst, yval    = catenc(ytrn, ytst, yval)



    ytrn, ytst, yval    = ohenc(ytrn, ytst, yval)
 
    # # print(xtrn[0])   # treebank (61014, 44232)   # brown (860100, 188)
    # # print(ytrn[0])   # treebank (61014, 46)      # brown (860100, 9)

    # # """
    # # ++++++++++++++++++++++++++++++++++++++++++
    # # MODEL
    # # """
    model_params = {
        'build_fn': build_model,
        'input_dim': xtrn.shape[1],
        'hidden_neurons': 512,
        'output_dim': ytrn.shape[1],
        'epochs': 3,
        'batch_size': 1024,
        'verbose': 1,
        'validation_data': (xval, yval),
        'shuffle': True
    }

    m = KerasClassifier(**model_params)
    hist = m.fit(xtrn, ytrn)
    score = m.score(xtst, ytst)
    print("score")
    print(score)
    m.model.save('model')


    #########
    # LOAD SAVED MODEL
    # m = keras.models.load_model('model')
    # ynew = m.predict_classes(xval)
    # print(ynew)
    # for i in range(len(xval)):
    #     print("X=%s, Predicted=%s, Ground=%s" % (dict_encoder.inverse_transform([xval[i]])[0], label_encoder.inverse_transform([ynew[i]]), label_encoder.inverse_transform([yval[i]])))


main()

