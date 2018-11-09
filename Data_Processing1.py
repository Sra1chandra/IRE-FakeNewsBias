import numpy as np
import os
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input
from keras.layers import Embedding
from sklearn.model_selection import KFold
from keras.models import load_model
from keras.utils import to_categorical

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

def getTextFromDocs(DirName,NewsType,FileName):
    f = open(DirName+FileName, 'r')
    text = f.read()
    f.close()
    text = text.split('\n')
    # print len(text)
    Input = []
    BiasLabel=[]
    Label=[]
    # print text[-1]
    for data in text[:-1]:
        file_name=re.match(r'.*?\.json',data,re.DOTALL).group(0)
        Input.append(load_doc(DirName+NewsType+'/'+file_name))
        bias_name=re.sub(r'.*?\.json',' ',data)
        BiasLabel.append(bias_name.strip())
        Label.append(NewsType)
        # print bias_name.strip()
        #     
    return Input,Label,BiasLabel

def ProcessLabel(Labels):
    FinalLabels=[]
    for label in Labels:
        if(label=='FakeNewsContent'):
            FinalLabels.append([0])
        else:
            FinalLabels.append([1])
    return FinalLabels

def ProcessBiasLabel(Labels):
    FinalLabels=[]
    for label in Labels:
        if(label=="QUESTIONABLE SOURCE"):
            FinalLabels.append([0,0,1,0])
        elif(label=="LEFT BIAS"):
            FinalLabels.append([0,1,0,0])
        elif(label=="LEFT-CENTER BIAS"):
            FinalLabels.append([0,1,1,0])
        elif(label=="RIGHT-CENTER BIAS"):
            FinalLabels.append([1,0,0,0])
        elif(label=="SATIRE"):
            FinalLabels.append([1,1,0,0])
        elif(label=="RIGHT BIAS"):
            FinalLabels.append([1,0,1,0])
        elif(label=="CONSPIRACY-PSEUDOSCIENCE"):
            FinalLabels.append([1,1,1,0])
        else:
            FinalLabels.append([0,0,0,0])

    return FinalLabels

def main():

    DirName  = './FakeNewsNet/Data/PolitiFact/'
    NewsType = 'FakeNewsContent'
    FileName = 'FakeNewsBias.txt' 
    FakeInput,FakeLabel,FakeBiasLabel=getTextFromDocs(DirName,NewsType,FileName)
    FakeLabel=ProcessLabel(FakeLabel)
    FakeBiasLabel=ProcessBiasLabel(FakeBiasLabel)
    
    NewsType = 'RealNewsContent'
    FileName = 'RealNewsBias.txt' 
    RealInput,RealLabel,RealBiasLabel=getTextFromDocs(DirName,NewsType,FileName)
    RealLabel=ProcessLabel(RealLabel)
    RealBiasLabel=ProcessBiasLabel(RealBiasLabel)

    Docs = FakeInput+RealInput
    Labels = FakeLabel+RealLabel
    BiasLabels = FakeBiasLabel+RealBiasLabel
    # print BiasLabels

    # create the tokenizer
    t = Tokenizer(lower=True)

    # fit the tokenizer on the documents
    t.fit_on_texts(Docs)

    # integer encode sequences of words
    sequences = t.texts_to_sequences(Docs)
    # print sequences

    # vocabulary size
    vocab_size = len(t.word_index) + 1

    seq_length=1000

    # pad sequence
    padded = pad_sequences(sequences,padding='post',maxlen=seq_length)
    # print(padded)

    # X input sequences 
    # y output
    InputText=np.array(padded)
    out1=(np.array(Labels))
    # print out1
    out2=np.array(BiasLabels)

    ##############################################################################################

    # # define model
    # model = Sequential()
    # model.add(Embedding(vocab_size, 500, input_length=seq_length))
    # # model.add(LSTM(100, return_sequences=True))
    # model.add(LSTM(100))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # # print(model.summary())

    # # compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(InputText, out2, batch_size=16, validation_split=0.1, epochs=3)

    #############################################################################################

    # # define model
    # model = Sequential()
    # model.add(Embedding(vocab_size, 500, input_length=seq_length))
    # # model.add(LSTM(100, return_sequences=True))
    # model.add(LSTM(100))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # # print(model.summary())

    # # compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(InputText, out1, batch_size=16, validation_split=0.1, epochs=3)

    #############################################################################################


    # model=Sequential()
    inp=Input(shape=(seq_length,))
    i1=Embedding(vocab_size,500,input_length=seq_length)(inp)
    i2=Embedding(vocab_size,500,input_length=seq_length)(inp)
    shared = LSTM(100)
    x1_layer=shared(i1)
    x2_layer=shared(i2)
    y1_layer=Dense(100, activation='relu')(x1_layer)
    y2_layer=Dense(100, activation='relu')(x2_layer)
    out_layer1 = Dense(1,activation='sigmoid', name="Real_output")(y1_layer)
    out_layer2 = Dense(4,activation='softmax', name="Bias_output")(y2_layer)
    model = Model(inputs=inp,outputs=[out_layer1,out_layer2])
    losses = {
	"Real_output": "binary_crossentropy",
	"Bias_output": "categorical_crossentropy",
    }  
    model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
    model.fit(InputText, [out1,out2], batch_size=16, validation_split=0.1, epochs=3)

    # kfold = KFold(5, True, 1)
    # max_accuracy=0
    # for train, test in kfold.split(X):       
    #     # fit model
    #     model.fit(X[train], y[train], batch_size=16, validation_split=0.1, epochs=3)
    #     score,accuracy = model.evaluate(X[test],y[test], batch_size=None, verbose=1)
    #     if(accuracy>max_accuracy):
    #         # save the model to file
    #         model.save('model.h5')
    #         max_accuracy=accuracy

    #     print accuracy
    
    # save the model to file
    # model.save('model.h5')
    # save the tokenizer
    # dump(t, open('tokenizer.pkl', 'wb'))

    # model=load_model('model.h5')
    # print model.evaluate(X,Y)

if __name__ == '__main__':
    main()
