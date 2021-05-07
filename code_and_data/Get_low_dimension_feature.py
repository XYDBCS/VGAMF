import  numpy as np

def read_txt(path):
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


def updating_U (W, A, U, V, lam):
    m, n = U.shape
    fenzi = (W*A).dot(V.T)
    fenmu = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))

    # fenmu = (W*(U.dot(V))).dot((V.T)) + lam*U
    U_new = U
    for i in range(m):
        for j in range(n):
            U_new[i,j] = U[i, j]*(fenzi[i,j]/fenmu[i, j])
    return U_new


def updating_V (W, A, U, V, lam):
        m,n = V.shape
        fenzi = (U.T).dot(W*A)
        fenmu = (U.T).dot(W*(U.dot(V)))+(lam/2)*(np.ones([m,n]))
        # fenmu = (U.T).dot(W*(U.dot(V)))+lam*V
        V_new = V
        for i in range(m):
            for j in range(n):
                V_new[i,j] = V[i, j]*(fenzi[i,j]/fenmu[i,j])
        return V_new

def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            #print("the shape of Ui", U[i,:].shape, V[:,j].shape)
            sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord = 1, keepdims = False))
    return  sum_obj




def get_low_feature(k,lam, th, A):#k is the number elements in the features, lam is the parameter for adjusting, th is the threshold for coverage state
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100#miRNA
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100#disease
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 1000:
        i =i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)
        # obj_value1 = obj_value
        # obj_value = objective_function(A, A, U, V, lam)
        # diff = abs(obj_value1 - obj_value)
        # print("ite ", i, diff, obj_value1)
        #print("iter", i)
    #print(U)
    #print(V)
    return U, V.transpose()




