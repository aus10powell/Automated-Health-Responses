# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Merge, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Concatenate
from keras.models import Model
import argparse
import os
from utilities import data_helper
#from utilities import my_callbacks

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(([self.validation_data[0], self.validation_data[1]]))
        print (y_pred)
        recall_k = compute_recall_ks(y_pred[:,0])

        self.accs.append(recall_k[10][1]) # append the recall 1@10

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def main():

    parser = argparse.ArgumentParser()
    parser.register('type','bool',data_helper.str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--input_dir', type=str, default='./option3_dataset/', help='Input dir')
    parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
    parser.add_argument('--model_fname', type=str, default='model/dual_encoder_lstm_classifier.h5', help='Model filename')
    #parser.add_argument('--embedding_file', type=str, default='embeddings/glove.840B.300d.txt', help='Embedding filename')
    parser.add_argument('--embedding_file', type=str, default='embeddings/glove.6B.300d.txt', help='Embedding filename') #
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    np.random.seed(args.seed)

    print("Starting...")

    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Now indexing word vectors...')

    embeddings_index = {}
    f = open(args.embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()

    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_dir + 'params.pkl', 'rb'))

    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))

    print("Now loading embedding matrix...")
    num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words , args.emb_dim))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Now building dual encoder lstm model...")
    # define lstm encoder
    encoder = Sequential()
    encoder.add(Embedding(output_dim=args.emb_dim,
                            input_dim=MAX_NB_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=True))
    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee')
    encoder.add(LSTM(units=args.hidden_size))

    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee1')
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee2')
    # encode the context and the response
    context_branch = encoder(context_input)
    response_branch = encoder(response_input)
    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee3')
    concatenated = merge([context_branch, response_branch], mode='mul')
    #concatenated = Concatenate()
    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee4')
    out = Dense((1), activation = "sigmoid") (concatenated)
    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee5')
    dual_encoder = Model([context_input, response_input], out)
    print('HEEEeeeeeeeeeeeeeeeeeeeeeeee6')
    dual_encoder.compile(loss='binary_crossentropy',
                    optimizer=args.optimizer)

    print(encoder.summary())
    print(dual_encoder.summary())

    print("Now loading UDC data...")

    train_c, train_r, train_l = pickle.load(open(args.input_dir + 'train.pkl', 'rb'))
    test_c, test_r, test_l = pickle.load(open(args.input_dir + 'test.pkl', 'rb'))
    dev_c, dev_r, dev_l = pickle.load(open(args.input_dir + 'dev.pkl', 'rb'))

    print('Found %s training samples.' % len(train_c))
    print('Found %s dev samples.' % len(dev_c))
    print('Found %s test samples.' % len(test_c))

    print("Now training the model...")

    histories = Histories()

    bestAcc = 0.0
    patience = 0

    print("\tbatch_size={}, nb_epoch={}".format(args.batch_size, args.n_epochs))

    for ep in range(1, args.n_epochs):

        dual_encoder.fit([train_c, train_r], train_l,
                batch_size=args.batch_size, epochs=1, callbacks=[histories],
                validation_data=([dev_c, dev_r], dev_l), verbose=1)

        curAcc =  histories.accs[0]
        if curAcc >= bestAcc:
            bestAcc = curAcc
            patience = 0
        else:
            patience = patience + 1

        # classify the test set
        y_pred = dual_encoder.predict([test_c, test_r])

        print("Perform on test set after Epoch: " + str(ep) + "...!")
        recall_k = data_helper.compute_recall_ks(y_pred[:,0])

        # stop training the model when patience = 10
        if patience > 10:
            print("Early stopping at epoch: "+ str(ep))
            break

    if args.save_model:
        print("Now saving the model... at {}".format(args.model_fname))
        dual_encoder.save(args.model_fname)

if __name__ == "__main__":
    main()
