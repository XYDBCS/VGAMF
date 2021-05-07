import numpy as np
from data_preprocessing import *
from sklearn.model_selection import KFold
from utilize import *
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from model import GCNModelAE, GCNModelVAE
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from optimizer import OptimizerAE, OptimizerVAE
import time
from sklearn.metrics import roc_auc_score
from Get_low_dimension_feature import get_low_feature
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 48, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0 , 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'miRNA-disease', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
model_str = FLAGS.model
dataset_str = FLAGS.dataset

#paprameters
k1= 78
k2 =37
D = 90#MF dimension
#upload the database and print the shape of database
#A = read_txt1("./database/HMDD 2.0/m_d_associations.txt")
A = np.load("./database/HMDD3.2/miRNA-disease association.npy")
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))
x,y = A.shape
score_matrix = np.zeros([x, y])
# get the samples for all the pos and negative ones
samples = get_all_the_samples(A)

label_all = []
y_score_all = []

#cross validation
kf = KFold(n_splits=10, shuffle=True)
iter = 0 #control each iterator
sum_score = 0
for train_index, test_index in kf.split(samples):
    if iter < 11:
        iter = iter + 1
        train_samples = samples[train_index, :]
        test_samples = samples[test_index, :]
        # updata the adjacency matrix by deleting the test part data
        new_A = update_Adjacency_matrix(A, test_samples)
        print(sum(sum(new_A)))
        #geting the integrated similarity matrix for miRNA and disease, k1 is the KNN for miRNA, k2 is the KNN for disease
        sim_m, sim_d = get_syn_sim(A, k1, k2)
        sim_m_0 = set_digo_zero(sim_m, 0)
        sim_d_0 = set_digo_zero(sim_d, 0)
        print("the maxmum of sim network",np.max(np.max(sim_m_0, axis = 0)), "the minimum of sim network", np.min(np.min(sim_m_0, axis=0)))
        print("the maxmum of simd network",np.max(np.max(sim_d_0, axis = 0)), "the minimum of simd network", np.min(np.min(sim_d_0, axis=0)))
        #getting features by adjacency matrix
        features_m = A
        features_d = A.transpose()
        #getting the feature extracting by VGAE on miRNA similarity network
        features_m = sp.coo_matrix(features_m)

        # Some preprocessing
        adj_norm = preprocess_graph(sim_m_0)
        # Define placeholders
        tf.disable_eager_execution()
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        num_nodes = sim_m.shape[0]

        features = sparse_to_tuple(features_m.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        # Create model
        model = None
        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
        pos_weight = 25
        #pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        #norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)# shape[0] = 913, adj.sum() = 10734
        norm = 0.5
        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_val = []
        acc_val = []
        val_roc_score = []

        sim_m_0 = sp.coo_matrix(sim_m_0)
        sim_m_0.eliminate_zeros()
        adj_label = sim_m_0 + sp.eye(sim_m_0.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
        print("Optimization Finished!")

        #getting the feature vectors for miRNAs
        # Z = sess.run([model.z], feed_dict=feed_dict)
        Z = sess.run(model.z, feed_dict=feed_dict)
        Z =np.array(Z)
        # Z =  np.reshape(Z,(-1,Z.shape[2]))
        feature_m = Z


 #training disease by VGAE
        #getting the feature extracting by VGAE on miRNA similarity network
        features_d = sp.coo_matrix(features_d)
        if FLAGS.features == 0:
            features = sp.identity(features.shape[0])  # featureless

        # Some preprocessing
        adj_norm = preprocess_graph(sim_d_0)
        num_nodes = sim_d.shape[0]

        features = sparse_to_tuple(features_d.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        # Create model
        model = None

        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
        pos_weight = 25
        norm = 0.5
        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_val = []
        acc_val = []
        val_roc_score = []

        sim_d_0 = sp.coo_matrix(sim_d_0)
        sim_d_0.eliminate_zeros()
        adj_label = sim_d_0 + sp.eye(sim_d_0.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
        print("Optimization Finished!")
        #getting the feature vectors for miRNAs
        # Z = sess.run([model.z], feed_dict=feed_dict)
        Z = sess.run(model.z, feed_dict=feed_dict)
        Z =np.array(Z)
        # Z =  np.reshape(Z,(-1,Z.shape[2]))
        print(Z.shape)
        feature_d = Z

        feature_MFm, feature_MFd = get_low_feature(D, 0.01, pow(10, -4), A)

        #emerge the miRNA feature and disease feature
        vect_len1 = feature_m.shape[1]
        vect_len2 = feature_d.shape[1]
        train_n = train_samples.shape[0]
        train_feature = np.zeros([train_n, 2*vect_len1+2*D])
        # train_feature = np.zeros([train_n, vect_len1+vect_len2])
        train_label = np.zeros([train_n])
        for i in range(train_n):
            train_feature[i,0:vect_len1] = feature_m[train_samples[i,0],:]
            train_feature[i,vect_len1 :(vect_len1+vect_len2)] = feature_d[train_samples[i,1], :]
            train_feature[i,(vect_len1+vect_len2):(vect_len1+vect_len2+D)] = feature_MFm[train_samples[i,0],:]
            train_feature[i, (vect_len1+vect_len2+D):(vect_len1+vect_len2+2*D)] = feature_MFd[train_samples[i,1],:]
            train_label[i] = train_samples[i,2]


        # get the featrue vectors of test samples
        test_N = test_samples.shape[0]
        test_feature = np.zeros([test_N,2*vect_len1+2*D])
        # test_feature = np.zeros([test_N,vect_len1+vect_len2])
        test_label = np.zeros(test_N)
        for i in range(test_N):
            test_feature[i, 0:vect_len1] = feature_m[test_samples[i,0], :]
            test_feature[i, vect_len1:(vect_len1+vect_len2)] =  feature_d[test_samples[i,1], :]
            test_feature[i, (vect_len1+vect_len2):(vect_len1+vect_len2+D)] = feature_MFm[test_samples[i,0],:]
            test_feature[i, (vect_len1+vect_len2+D):(vect_len1+vect_len2+2*D)] = feature_MFd[test_samples[i,1], :]
            test_label[i]=test_samples[i,2]

#train the neural network model
        model = BuildModel(train_feature, train_label)
        #model = BuildModel1(train_feature1, train_feature2, train_label)
        y_score = np.zeros(test_N)
        y_score = model.predict(test_feature)[:,0]

        #y_score = model.predict([test_feature1, test_feature2])[:,0]
        print('y_scores', y_score.shape)
        roc_score = roc_auc_score(test_label, y_score)
        print('origin-roc',roc_score, iter)
        sum_score = sum_score+roc_score
        test_label = test_label.tolist ()
        y_score = y_score.tolist()
        label_all.append(test_label)
        y_score_all.append(y_score)
#calculate the ROC value
label_all = [i for item in label_all for i in item]
y_score_all = [i for item in y_score_all for i in item]
label_all = np.array(label_all)
y_score_all = np.array(y_score_all)
print("labe.shape", label_all.shape)
print("ave", sum_score/10)
np.save('5_fold_add_dot_label_multi.npy', label_all)
np.save('5_fold_add_dot_score_multi.npy', y_score_all)





