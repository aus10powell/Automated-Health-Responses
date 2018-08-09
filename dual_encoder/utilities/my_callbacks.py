from __future__ import division
import keras
import numpy as np
from data_helper import compute_recall_ks


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

    #def compute_recall_ks(self, probas):
        #recall_k = {}
        #for group_size in [2, 5, 10]:
            #recall_k[group_size] = {}
            #print ('group_size: %d' % group_size)
            #for k in [1, 2, 5]:
                #if k < group_size:
                    #recall_k[group_size][k] = self.recall(probas, k, group_size)
                    #print ('recall@%d' % k, recall_k[group_size][k])
        #return recall_k

    #def recall(self, probas, k, group_size):
        #test_size = 10
        #n_batches = len(probas) // test_size
        #n_correct = 0
        #for i in range(n_batches):
            #batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            #indices = np.argpartition(batch, -k)[-k:]
            #if 0 in indices:
                #n_correct += 1
        #return n_correct / (len(probas) / test_size)
