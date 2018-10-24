import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import KFold
from keras.models import load_model

# from pickle import dump

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()

	return text

def getTextFromDocs(dir_name):
    file_names = os.listdir(dir_name)
    # file_names=['BuzzFeed_Fake_1-Webpage.json']
    text_data=[]
    for file_name in file_names:
        text_data.append(load_doc(dir_name+file_name))
    
    return text_data

def main():

    dir_name='./FakeNewsNet/Data/PolitiFact/FakeNewsContent/'
    Fakedocs=getTextFromDocs(dir_name)
    label_Fake=[0]*len(Fakedocs)

    dir_name='./FakeNewsNet/Data/PolitiFact/RealNewsContent/'
    Realdocs=getTextFromDocs(dir_name)
    label_Real=[1]*len(Fakedocs)

    docs=Fakedocs+Realdocs
    labels=label_Fake+label_Real

    # create the tokenizer
    t = Tokenizer(lower=True)

    # fit the tokenizer on the documents
    t.fit_on_texts(docs)

    # integer encode sequences of words
    sequences = t.texts_to_sequences(docs)
    # print sequences

    # vocabulary size
    vocab_size = len(t.word_index) + 1

    seq_length=1000

    # pad sequence
    padded = pad_sequences(sequences,padding='post',maxlen=seq_length)
    # print(padded)

    # X input sequences 
    # y output
    X1=np.array(padded)
    y1=np.array(labels)
    order = np.arange(len(X1))
    X=X1
    y=y1

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 500, input_length=seq_length))
    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # print(model.summary())

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    kfold = KFold(5, True, 1)
    max_accuracy=0
    for train, test in kfold.split(X):       
        # fit model
        model.fit(X[train], y[train], batch_size=16, validation_split=0.1, epochs=3)
        score,accuracy = model.evaluate(X[test],y[test], batch_size=None, verbose=1)
        if(accuracy>max_accuracy):
            # save the model to file
            model.save('model.h5')
            max_accuracy=accuracy

        print accuracy
    
    # save the model to file
    # model.save('model.h5')
    # save the tokenizer
    # dump(t, open('tokenizer.pkl', 'wb'))

    model=load_model('model.h5')
    print model.evaluate(X,Y)

if __name__ == '__main__':
    main()
