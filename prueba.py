import numpy as np
from lineal_reg import *

TRAINING_SET_SIZE = 200

x = np.linspace(-10, 30, TRAINING_SET_SIZE)

X = np.vstack(
    (
        np.ones(TRAINING_SET_SIZE),
        x,
        x ** 2,
        x ** 3,
    )
).T

y = (5 + 2 * x ** 3 + np.random.randint(-9000, 9000, TRAINING_SET_SIZE)).reshape(
    TRAINING_SET_SIZE,
    1
)

m, n = X.shape
theta_0 = np.random.rand(n, 1)



clean_data = np.genfromtxt('kc_house_data.csv', delimiter=',',usecols=(2,5), skip_header=True)


np.random.seed(13)
np.random.shuffle(clean_data)

data = clean_data.copy()


X = data[:, 1].reshape(-1,1)
Y = data[:, 0].reshape(-1,1)







x_train = X[:int(0.6*len(X))]
x_test = X[int(0.6*len(X)):]
y_train = Y[:int(0.6*len(X))]
y_test = Y[int(0.6*len(X)):]


train_data = data[:int(0.6*len(data))]
test_data = data[int(0.6*len(data)):]

theta_0 = np.zeros((x_train.shape[1], 1))
h = (x_train@theta_0)

print(x_train.shape)

#print(x_train.shape)
#print(theta_0.shape)
#print(h.shape)
#print(y_train.shape)
#print((h-y_train).shape)
#(m, n) (n, 1)
#(n,m) (1,m)


theta, costs, thetas = Descent(x_train, y_train, theta_0)

y_pred = x_test @ theta

def SE(test, y):
    err = test-y
    print(err)
    return np.sum(err**2)

def RMSE(test, y):
    se = SE(test,y)
    print(se)
    return np.sqrt((1/len(test))*se)

def RAE(test,y):
    mean = np.mean(y)
    num = np.sum(abs(test-y))
    den = np.sum(abs(mean-y))
    
    return num/den

def RRAE(test,y):
    mean = np.mean(y)
    num = np.sum((test-y)**2)
    den = np.sum((mean-y)**2)
    

    
    return np.sqrt(num/den)

def PolynomialFeature(arr, degree):
    return arr**degree

print("Accuracy: ", RRAE(y_pred, y_test))





