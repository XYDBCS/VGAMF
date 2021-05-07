import numpy as np
import math
import random
# the function of GIP kernel

def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix
def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def read_txt1(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split( )
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data


#W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn


#updataing rules
def MiRNA_updating (S1,S2,S3,S4, P1,P2,P3,P4):
    it = 0
    P = (P1+P2+P3+P4)/4
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2+P3+P4)/3),S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot (np.dot(S2,(P1+P3+P4)/3),S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot (np.dot(S3,(P1+P2+P4)/3),S3.T)
        P333 = new_normalization(P333)
        P444 = np.dot(np.dot(S4,(P1+P2+P3)/3),S4.T)
        P444 = new_normalization(P444)
        P1 = P111
        P2 = P222
        P3 = P333
        P4 = P444
        P_New = (P1+P2+P3+P4)/4
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def disease_updating(S1,S2, P1,P2):
    it = 0
    P = (P1+P2)/2
    dif = 1
    while dif> 0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,P2),S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot (np.dot(S2,P1),S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1+P2)/2
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P

def get_syn_sim (A, k1, k2):
    # disease_sim1 = read_txt1("./database/HMDD 2.0/disease_semantic_similarity.txt")
    # miRNA_sim1 = read_txt1("./database/HMDD 2.0/miRNA_functional_similarity.txt")
    # miRNA_sim2 = read_txt1 ("./database/HMDD 2.0/miRNA_sequence_similarity.txt")
    # miRNA_sim3 = read_txt1("./database/HMDD 2.0/miRNA_semantic_similarity.txt")

    disease_sim1 = read_txt1("./database/HMDD3.2/disease_semantic_sim.txt")
    miRNA_sim1 = read_txt1("./database/HMDD3.2/miRNA_functional_sim.txt")
    miRNA_sim2 = read_txt1 ("./database/HMDD3.2/miRNA_sequence_sim.txt")
    miRNA_sim3 = read_txt1("./database/HMDD3.2/miRNA_semantic_sim.txt")

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)
    #miRNA_sim1 = GIP_m_sim
    m1 = new_normalization(miRNA_sim1)
    # m1 = miRNA_sim1
    # m2 = miRNA_sim2
    # m3 = miRNA_sim3
    # m4 = GIP_m_sim
    m2 = new_normalization(miRNA_sim2)
    m3 = new_normalization(miRNA_sim3)
    m4 = new_normalization(GIP_m_sim)
    Sm_1 = KNN_kernel(miRNA_sim1, k1)
    Sm_2 = KNN_kernel(miRNA_sim2, k1)
    Sm_3 = KNN_kernel(miRNA_sim3, k1)
    Sm_4 = KNN_kernel(GIP_m_sim, k1)
    Pm = MiRNA_updating(Sm_1,Sm_2,Sm_3,Sm_4, m1, m2, m3,m4)
    Pm_final = (Pm + Pm.T)/2
    #np.save('m_sim_final.npy', Pm_final)
    #np.save('m_sim_final.txt', Pm_final)
    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(GIP_d_sim)
    #d1 = disease_sim1
    #d2 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating(Sd_1,Sd_2, d1, d2)
    Pd_final = (Pd+Pd.T)/2
    #np.save('d_sim_final.npy', Pd_final)
    #np.save('d_sim_final.txt', Pd_final)

    return Pm_final, Pd_final

def get_all_the_samples(A):
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i,j] ==1:
                pos.append([i,j,1])
            else:
                neg.append([i,j,0])
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples

def get_all_the_samples1(A):
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i,j] ==1:
                pos.append([i,j,1])
            else:
                neg.append([i,j,0])
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    neg = np.array(neg)
    return samples, neg

def update_Adjacency_matrix (A, test_samples):
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i,2] ==1:
            A_tep [test_samples[i,0], test_samples[i,1]] = 0
    return A_tep

def set_digo_zero(sim, z):
    sim_new = sim.copy()
    n = sim.shape[0]
    for i in range(n):
        sim_new[i][i] = z
    return sim_new

#get the sparse similarity matrix, k is the first k nearest neibors

def similarity_spare (A, k):
    m = A.shape[0]
    S = np.zeros([m,k])
    for i in range(m):
        tep = np.argsort(-A[i,:])
        S[i,0:k] = tep[0:k]

    W = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i in S[j,:] and j in S[i,:]:
                W[i,j] = 1
            elif i in S[j,:] or j in S[i,:]:
                W[i,j] = 0.5
            else:
                W[i,j] = 0
    A = np.multiply(A, W)
    return A


















