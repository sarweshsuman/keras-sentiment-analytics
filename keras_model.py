from keras.layers import Dense, Dropout, Activation , LSTM, Bidirectional
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
import keras

def create_model(vocab_size,embedding_size,max_sentence_length,dropout,embedding_matrix=None):
	model = Sequential()
	if embedding_matrix is None:
		model.add(Embedding(vocab_size+1,embedding_size,input_length=max_sentence_length))
	else:
		model.add(Embedding(vocab_size+1,embedding_size,weights=[embedding_matrix],input_length=max_sentence_length,trainable=False))
	model.add(Bidirectional(LSTM(64,return_sequences=True),merge_mode='mul'))
	model.add(Dropout(dropout))
	model.add(Bidirectional(LSTM(128,return_sequences=True),merge_mode='mul'))
	model.add(Dropout(dropout))
	model.add(Bidirectional(LSTM(256)))
	model.add(Dense(2,activation='sigmoid'))
	adam = Adam(lr=0.0001, decay=1e-5)
	model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['mse', 'acc'])
	return model
