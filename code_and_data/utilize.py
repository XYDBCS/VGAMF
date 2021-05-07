import numpy as np
import random
from keras.layers import Dense, Input, dot
from keras.models import Model
def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split(',')
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data

#get the features of each node by integrating three different network
def get_feature(M_M, M_D,D_D):
    H1 = np.hstack(( M_M, M_D))
    H2 = np.hstack((M_D.transpose(),D_D))
    H = np.vstack((H1,H2))
    print('The shape of H', H.shape)
    return H

#find the miRNA-disease part
def find_mi_D (edges):
    m_d = []
    for i in range(edges.shape[0]):
        if edges[i,0]<577 and edges[i,1] >576:
            m_d.append(edges[i,:])
        elif edges[i,0]>576 and edges[i,1]<577:
            m_d.append(edges[i,:])
    return m_d

def BuildModel(train_x,train_y):
    # This returns a tensor
    l = len(train_x[1])
    inputs = Input(shape=(l,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y)  # starts training
    return model


def BuildModel1(train_x1,train_x2,train_y):
    # This returns a tensor
    inputs1 = Input(shape=(train_x1.shape[1],))
    inputs2 = Input(shape=(train_x2.shape[1],))
    # a layer instance is callable on a tensor, and returns a tensor
    x1 = Dense(64, activation='relu')(inputs1)
    x1 = Dense(32, activation='relu')(x1)
    x2 = Dense(64, activation='relu')(inputs2)
    x2 = Dense(32, activation='relu')(x2)
    #x = Dense(16, activation='relu')(x)
    predictions = dot([inputs1,inputs2], axes=1, normalize=True)
    # predictions = np.dot([inputs1,inputs2])
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs1,inputs2], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit([train_x1,train_x2], train_y)  # starts training
    return model

def BuildModel(train_x,train_y):
    # This returns a tensor
    l = len(train_x[1])
    inputs = Input(shape=(l,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y)  # starts training
    return model



