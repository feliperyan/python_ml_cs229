# Stanford's Machine Learning by Andrew Ng - Assignments in Python format
# Felipe Ryan 2014

# The following import forces floatpoint division so 5/2 = 2.5 and not 2
from __future__ import division
import numpy as np
import math
from scipy.optimize import minimize
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm


# Helper function
def sigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))


# Applies function above to every element:
def VectorizedSigmoid(matrix):
    #print 'vec sig matrix: ' + str(matrix)
    vs = np.vectorize(sigmoid)
    return vs(matrix)


# Helper function
def score(X, y, theta):
    ssres = np.sum((X.dot(theta.T) - y.T) ** 2)
    sstot = np.sum((y.T - np.mean(y)) ** 2)

    return 1 - (ssres / sstot)


# Feature Normalization
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = np.subtract(X, mu)
    sigma = np.std(X, axis=0, ddof=1)
    return np.divide(X_norm, sigma)


# 100% done and tested
def computeCostMulti(X, y, theta):
    m = float(X.shape[0])
    J = 0
    J = np.sum(np.power(X.dot(theta) - y, 2)) / (2 * m)
    return J


# 100% done and tested
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = float(X.shape[0])
    J_hist = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - ((alpha / m) * (X.T.dot(h - y)))

        J_hist[i] = computeCostMulti(X, y, theta)

    return theta, J_hist


# WEEK 4 - 1/4 Regularised Logistic Regression
def lrCostFunction(theta, X, y, theLambda):
    print ('lrCost: ' + str(theta.shape))
    m = X.shape[0]
    J = 0

    grad = np.zeros(theta.shape)

    J = ((-y.T.dot(np.log(VectorizedSigmoid(X.dot(theta))))) -
        ((1 - y).T.dot(np.log(1 - VectorizedSigmoid(X.dot(theta)))))) / float(m)

    J = J + (float(theLambda) / (2 * m)) * np.sum(np.power(theta[1:, :], 2))

    grad = (1.0 / m) * (X.T.dot((VectorizedSigmoid(X.dot(theta)) - y)))

    # Following line does the regularisation:
    grad[1:, :] = grad[1:, :] + (theta[1:, :] * (float(theLambda) / y.shape[0]))

    print ('lrCost grad: ' + str(grad.shape))

    return J, grad


def gradientDescentMultiLogistic(X, y, theta, alpha, theLambda, num_iters):
    J_hist = np.zeros((num_iters, 1))

    for i in range(num_iters):
        (J, th) = lrCostFunction(theta, X, y, theLambda)
        theta = theta - (alpha * th)
        J_hist[i] = J

    return theta, J_hist


# WEEK 4 - 2/4 One-vs-All classifier Training
def oneVsAll(X, y, num_labels, theLambda=0.1, alpha=0.1, num_iters=50):
    n = X.shape[1]

    all_theta = np.zeros((num_labels, n))

    for i in range(num_labels):

        new_y = y == i
        new_y = new_y.astype(int)

        temp_theta = np.zeros((n, 1))

        (th, j) = gradientDescentMultiLogistic(X, new_y, temp_theta, alpha, theLambda, num_iters)

        all_theta[i] = th.T

    return all_theta


# WEEK 4 - 3/4 One-vs-All Classifier prediction
def predictOneVsAll(all_theta, X):
    probs = X.dot(all_theta.T)
    predictions = np.argmax(probs, axis=1)

    return predictions


# WEEK 5 - 3/5 Sigmoid Gradient
def sigmoidGrad(val):
    return (1.0 / (1.0 + math.exp(-val))) * \
        (1 - (1.0 / (1.0 + math.exp(-val))))


# Applies function above to every element:
def VectorizedSigmoidGrad(matrix):
    vs = np.vectorize(sigmoidGrad)
    return vs(matrix)


# Helper function for the generalised nnCostFunction
def getWeightsFromFlatData(nn_params, layers):
    input_layer = layers.pop(0)
    thetas = []
    
    t = nn_params[0:(layers[0] * (input_layer + 1))]
    t = t.reshape((layers[0], (input_layer + 1)), order='F')
    thetas.append(t)

    for i in range(len(layers)-1):
        start = thetas[i].shape[0] * thetas[i].shape[1]
        end = start + (layers[i+1] * (layers[i] + 1))
        tt = nn_params[start:end]
        tt = tt.reshape((layers[i+1], layers[i]+1), order='F')
        thetas.append(tt)

    return thetas

# Need to further generalise this to accept multiple layers
def nnCostFunctionGeneralised(nn_params, layers, X, y, theLambda=0):
    J = 0
    m = X.shape[0]
    thetas = []

    if(len(layers) < 2):
        print ('Define more layers')
        return False, False

    thetas = getWeightsFromFlatData(nn_params, layers)
    t1 = thetas[0]
    t2 = thetas[1]

    hidden = X.dot(t1.T)
    hidden = VectorizedSigmoid(hidden)
    
    # Add bias term:
    hidden = np.insert(arr=hidden, obj=0, values=1, axis=1)

    output = hidden.dot(t2.T)
    output = VectorizedSigmoid(output)
    
    # Building a matrix yy representing y such that each row of yy consists of 10
    # columns and has a value of 1 in the corresponding column to the value of y for the 
    # same row. Ie, if the three first values of y are: 3, 9, 5 then the first 3 rows of yy are:
    # 0 0 1 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 1 0
    # 0 0 0 0 1 0 0 0 0 0
    yy = np.zeros((m, num_labels))
    for i in range(y.shape[0]):
        yy[i, (y[i] - 1)] = 1

    # Computing Cost...
    for i in range(y.shape[0]):
        J += yy[i].dot(np.nan_to_num( np.log(output[i]) ).T) + \
            (1 - yy[i]).dot(np.nan_to_num(np.log(1 - output[i])).T)

    J = J * (-1 / m)
    
    # Regularization:
    t1r = t1[:, 1:]
    t2r = t2[:, 1:]

    sumt1r = np.sum(np.sum(np.power(t1r, 2), axis=1))
    sumt2r = np.sum(np.sum(np.power(t2r, 2), axis=1))

    reg = (sumt1r + sumt2r) * (theLambda / (2 * m))

    J += reg
    
    # Computing gradient
    delta3 = output - yy
    # Feed forward for the hidden layer same as above:
    z2 = X.dot(t1.T)
    z2 = np.insert(arr=z2, obj=0, values=1, axis=1)
    
    # Getting the gradient for the hidden layer's thetas:
    delta2 = np.multiply(delta3.dot(t2), (VectorizedSigmoidGrad(z2)))[:, 1:]
    
    d1 = delta2.T.dot(X)
    d2 = delta3.T.dot(hidden)

    # Regularization, skipping the bias term...!
    d1 = (d1 / m) + ((theLambda / m) * (np.insert(arr=t1[:, 1:], obj=0, values=0, axis=1)))
    d2 = (d2 / m) + ((theLambda / m) * (np.insert(arr=t2[:, 1:], obj=0, values=0, axis=1)))

    grad = np.concatenate((d1.flatten('F'), d2.flatten('F')))

    return grad, J


# WEEK 5 - 1/5 Feedforward and Cost Function
# WEEK 5 - 2/5 Regularised Cost Function
# WEEK 5 - 4/5 Neural Net Gradient Function (Backpropagation)
# WEEK 5 - 5/5 Regularised Gradient
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):
    J = 0
    m = X.shape[0]
    
    # scipy.optimize minimize expects the thetas as one long vector, so I had to rebuild it here. 
    t1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))]
    t1 = t1.reshape((hidden_layer_size, (input_layer_size + 1)), order='F')

    t2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    t2 = t2.reshape((num_labels, (hidden_layer_size + 1)), order='F')
    
    # Activating inputs and computing z
    hidden = X.dot(t1.T)
    hidden = VectorizedSigmoid(hidden)
    
    # Add bias term:
    hidden = np.insert(arr=hidden, obj=0, values=1, axis=1)

    output = hidden.dot(t2.T)
    output = VectorizedSigmoid(output)
    
    # Building a matrix yy representing y such that each row of yy consists of 10
    # columns and has a value of 1 in the corresponding column to the value of y for the 
    # same row. Ie, if the three first values of y are: 3, 9, 5 then the first 3 rows of yy are:
    # 0 0 1 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 1 0
    # 0 0 0 0 1 0 0 0 0 0
    yy = np.zeros((m, num_labels))
    for i in range(y.shape[0]):
        yy[i, (y[i] - 1)] = 1

    # Computing Cost...
    for i in range(y.shape[0]):
        J += yy[i].dot(np.nan_to_num( np.log(output[i]) ).T) + \
            (1 - yy[i]).dot(np.nan_to_num(np.log(1 - output[i])).T)

    J = J * (-1 / m)
    
    # Regularization:
    t1r = t1[:, 1:]
    t2r = t2[:, 1:]

    sumt1r = np.sum(np.sum(np.power(t1r, 2), axis=1))
    sumt2r = np.sum(np.sum(np.power(t2r, 2), axis=1))

    reg = (sumt1r + sumt2r) * (theLambda / (2 * m))

    J += reg
    
    # Computing gradient
    delta3 = output - yy
    # Feed forward for the hidden layer same as above:
    z2 = X.dot(t1.T)
    z2 = np.insert(arr=z2, obj=0, values=1, axis=1)
    
    # Getting the gradient for the hidden layer's thetas:
    delta2 = np.multiply(delta3.dot(t2), (VectorizedSigmoidGrad(z2)))[:, 1:]
    
    d1 = delta2.T.dot(X)
    d2 = delta3.T.dot(hidden)

    # Regularization, skipping the bias term...!
    d1 = (d1 / m) + ((theLambda / m) * (np.insert(arr=t1[:, 1:], obj=0, values=0, axis=1)))
    d2 = (d2 / m) + ((theLambda / m) * (np.insert(arr=t2[:, 1:], obj=0, values=0, axis=1)))

    grad = np.concatenate((d1.flatten('F'), d2.flatten('F')))

    return grad, J


def randInitialiseWeights(l_in, l_out, epsilon):
    w = np.random.random((l_out, l_in))
    return w * 2 * epsilon - epsilon


# Testing out gradient descent instead of the smart minimizing function - please ignore:
def gradientDescentNeuralNetwork(theta, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda, alpha, num_iters):
    J_hist = np.zeros((num_iters, 1))

    old_alpha = alpha

    for i in range(num_iters):
        (th, J) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
        theta = theta - (alpha * th)
        J_hist[i] = J

        # An attempt at a crude adaptive alpha (learning rate)
        if i > 1:
            if J < J_hist[i - 1]:
                alpha += old_alpha
            else:
                alpha = alpha / 2

        print ('Iter %d | alpha: %f | J = %f' % (num_iters - i, alpha, J))

    return theta, J_hist


# Helper function in an attempt to use fmin_cg (does work!)
def funCostNeuralNetwork(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):
    (gg, jj) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
    return jj


# Helper function in an attempt to use fmin_cg (does work!)
def funGradNeuralNetwork(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):
    (gg, jj) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
    return gg


# WEEK 4 - 4/4 Neural Network Prediction Function
def predict(t1, t2, X):
    h1 = VectorizedSigmoid(X.dot(t1.T))
    h1 = np.insert(arr=h1, obj=0, values=1, axis=1)
    h2 = VectorizedSigmoid(h1.dot(t2.T))

    p = h2.argmax(axis=1)

    p = p + 1

    return p


# Testing Week 5 code:
def testNeuralNetwork():
    # Load data
    d = sio.loadmat('data/ex4data1.mat')
    X = d['X']
    y = d['y']

    # Adding bias values
    X = np.insert(arr=X, obj=0, values=1, axis=1)

    args = (400, 25, 10, X, y, 1)

    i_t1 = randInitialiseWeights(401, 25, 0.12)
    i_t2 = randInitialiseWeights(26, 10, 0.12)

    i_nn_params = np.concatenate((i_t1.flatten(order='F'), i_t2.flatten(order='F')), axis=0)

    print ('Done reading in data, now training Neural Network...')

    def callbackFunc2(theta):
        (gg, jj) = nnCostFunction(theta, args[0], args[1], args[2], args[3], args[4], args[5])
        print ('Current cost: %f' % jj)

    res2 = minimize(funCostNeuralNetwork, i_nn_params, args=args, method='CG',
                    jac=funGradNeuralNetwork, options={'maxiter': 50}, callback=callbackFunc2)

    optT1 = res2['x'][0:(25 * (400 + 1))]
    optT1 = optT1.reshape((25, (400 + 1)), order='F')
    optT2 = res2['x'][(25 * (400 + 1)):]
    optT2 = optT2.reshape((10, (25 + 1)), order='F')

    p = np.asmatrix(predict(optT1, optT2, X)).T
    score = (p == y).astype(int).mean() * 100

    print (res2['message'])
    print ('Cost: %f' % res2['fun'])
    print ('Score: %0.2f%%' % score)


def testNeuralNetworkGeneralised():
    # Load data
    d = sio.loadmat('data/ex4data1.mat')
    X = d['X']
    y = d['y']

    # Adding bias values
    X = np.insert(arr=X, obj=0, values=1, axis=1)

    args = (400, 25, 10, X, y, 1)

    i_t1 = randInitialiseWeights(401, 25, 0.12)
    i_t2 = randInitialiseWeights(26, 10, 0.12)

    i_nn_params = np.concatenate((i_t1.flatten(order='F'), i_t2.flatten(order='F')), axis=0)

    print ('Done reading in data, now training Neural Network...')

    def callbackFunc2(theta):
        (gg, jj) = nnCostFunction(theta, args[0], args[1], args[2], args[3], args[4], args[5])
        print ('Current cost: %f' % jj)

    res2 = minimize(funCostNeuralNetwork, i_nn_params, args=args, method='CG',
                    jac=funGradNeuralNetwork, options={'maxiter': 50}, callback=callbackFunc2)

    optT1 = res2['x'][0:(25 * (400 + 1))]
    optT1 = optT1.reshape((25, (400 + 1)), order='F')
    optT2 = res2['x'][(25 * (400 + 1)):]
    optT2 = optT2.reshape((10, (25 + 1)), order='F')

    p = np.asmatrix(predict(optT1, optT2, X)).T
    score = (p == y).astype(int).mean() * 100

    print (res2['message'])
    print ('Cost: %f' % res2['fun'])
    print ('Score: %0.2f%%' % score)
    

# WEEK 6 - 1/5 - Regularised Linear Regression Cost Function
# WEEK 6 - 2/5 - Regularised Linear Regression Gradient
def linearRegCostFunction(X, y, theta, theLambda):
    m = X.shape[0]
    grad = np.zeros(X.shape[1])

    J = sum(np.power(((X.dot(theta)) - y), 2)) / (2 * m)
    J = J + (theLambda / float(2 * m)) * sum(np.power(theta[1:, ], 2))

    grad = X.T.dot(X.dot(theta) - y) / m
    reg = theta[1:, ] * (theLambda / m)
    grad[1:, ] += reg

    return J, grad


# Helper function for trainLinearReg
def lnRegCostFunction(t, X, y, theLambda):
    t = t.reshape((X.shape[1], 1), order='F')
    J, grad = linearRegCostFunction(X, y, t, theLambda)
    return J


# Helper function for trainLinearReg
def lnRegGradFunction(t, X, y, theLambda):
    t = t.reshape((X.shape[1], 1), order='F')
    J, grad = linearRegCostFunction(X, y, t, theLambda)
    return grad.flatten(order='F')


# Helper function originally provided by Andrew Ng, I translated it to Python
def trainLinearReg(X, y, theLambda):
    t = np.zeros((X.shape[1], 1))
    t = t.flatten(order='F')

    args = (X, y, theLambda)
    res1 = minimize(lnRegCostFunction, t, args=args, method='CG', jac=lnRegGradFunction, options={'maxiter': 200})

    return res1.x.reshape((res1.x.size, 1))  # so we get a vector


# WEEK 6 - 3/5 - Learning Curve
def learningCurve(X, y, Xval, yval, theLambda):
    m = X.shape[0]

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(m):
        theta = trainLinearReg(X[0:i + 1, :], y[0:i + 1, :], theLambda)

        J, grad = linearRegCostFunction(X[0:i + 1, :], y[0:i + 1, :], theta, theLambda)
        error_train[i] = J

        J, grad = linearRegCostFunction(Xval, yval, theta, theLambda)
        error_val[i] = J

    return error_train, error_val


# WEEK 6 - 5/5 - Cross Validation Curve
def validationCurve(X, y, Xval, yval):
    lambda_vec = np.matrix([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
    ev = np.zeros((lambda_vec.shape[0], 1))
    et = np.zeros((lambda_vec.shape[0], 1))

    for i in range(lambda_vec.size):
        theta = trainLinearReg(X, y, lambda_vec[i])
        (j, g) = linearRegCostFunction(X, y, theta, 0)
        et[i] = j
        (j, g) = linearRegCostFunction(Xval, yval, theta, 0)
        ev[i] = j

    return lambda_vec, et, ev


# WEEK 6 - 4/5 - Polynomial Feature Mapping
def polyFeatures(X, p):
    x_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        x_poly[:, i] = np.power(X, i+1)

    return x_poly


def testLinearRegression():
    import scipy.io as sio
    import matplotlib.pyplot as plt

    d = sio.loadmat('data/ex5data1.mat')
    X = d['X']
    y = d['y']
    # Adding bias values
    X = np.insert(arr=X, obj=0, values=1, axis=1)
    Xval = d['Xval']
    Xval = np.insert(arr=Xval, obj=0, values=1, axis=1)
    yval = d['yval']

    Xtest = d['Xtest']
    Xtest = np.insert(arr=Xtest, obj=0, values=1, axis=1)

    (et, ev) = learningCurve(X, y, Xval, yval, 0)

    print ('Training Error:\n')
    print (et)
    print ('\n')

    print ('Cross Validation Error:\n')
    print (ev)
    print ('\n')

    plt.plot(range(12), ev, range(12), et)
    plt.axis([0, 13, 0, 150])
    plt.ylabel('Error')
    plt.xlabel('Num of training examples')
    plt.legend(('Train', 'Cross Val'))
    plt.show()

    p = 8

    # Removed intercept next:
    X_poly = polyFeatures(X[:, 1], p)
    normalised_X_poly = featureNormalize(X_poly)
    # re-adding intercept
    normalised_X_poly = np.insert(arr=normalised_X_poly, obj=0, values=1, axis=1)

    print ('Normalised X_poly with p = 8, values should be similar to ex5')
    print (normalised_X_poly[0])

    Xtest_poly_normal = featureNormalize(polyFeatures(Xtest[:, 1], p))
    Xval_poly_normal = featureNormalize(polyFeatures(Xval[:, 1], p))

    Xtest_poly_normal = np.insert(arr=Xtest_poly_normal, obj=0, values=1, axis=1)
    Xval_poly_normal = np.insert(arr=Xval_poly_normal, obj=0, values=1, axis=1)

    theta = trainLinearReg(X_poly, y, 0)

    plt.plot(X, y, )
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')


# WEEK 7 - 1/4 - Gaussian Kernel
def gaussianKernel(x1, x2, sigma=0.3):
    x1Flat = x1.flatten(order='F')
    x2Flat = x2.flatten(order='F')

    return np.exp(-(np.sum(np.power((x1Flat - x2Flat), 2)) / float((2 * (sigma ** 2)))))


def dataset3params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    c_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # added a few options for sigma as it seemed to give me a better result
    # at around C=3 and sigma=55 in preliminary testing.
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40, 50, 60]
    J = list()
    cost = 9999999

    for c in c_vec:
        for s in sigma_vec:
            clf = svm.SVC(kernel='rbf', C=c, gamma=s)
            # ravel used to shut up sklearn warning
            clf.fit(X, y.ravel())
            pred = clf.predict(Xval)
            res = np.logical_xor(pred, yval.flatten()).astype(int)
            res = np.mean(res)

            J.append(res)
            newcost = res

            if newcost < cost:
                C = c
                sigma = s
                cost = newcost

    return C, sigma, J


def testSVM():

    def plotData(X, y):
        x1Pos = X[np.where(y == 1)[0], 0]
        x2Pos = X[np.where(y == 1)[0], 1]

        x1Neg = X[np.where(y == 0)[0], 0]
        x2Neg = X[np.where(y == 0)[0], 1]

        plt.plot(x1Pos, x2Pos, '+', x1Neg, x2Neg, '^')
        plt.show()

    def visualBoundaryLinear(X, y, model, title):
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.2
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        ZZ = Z.reshape(xx.shape)
        plt.contour(xx, yy, ZZ)

        x1Pos = X[np.where(y == 1)[0], 0]
        x2Pos = X[np.where(y == 1)[0], 1]

        x1Neg = X[np.where(y == 0)[0], 0]
        x2Neg = X[np.where(y == 0)[0], 1]

        plt.title(title)
        plt.plot(x1Pos, x2Pos, '+', x1Neg, x2Neg, '^')
        plt.show()

    d = sio.loadmat('data/ex6data1.mat')
    X = d['X']
    y = d['y']

    plotData(X, y)

    print ('\nUsing SKLearn\'s svm.LinearSVC to train a model. Drawing the decision boundary...')

    # Adding bias values
    #X = np.insert(arr=X, obj=0, values=1, axis=1)

    C = [0.1, 1, 10, 100, 1000, 10000]
    for i in C:
        clf = svm.LinearSVC(C=i)
        # ravel used to shut up sklearn warning
        clf.fit(X, y.ravel())
        visualBoundaryLinear(X, y, clf, 'Value of C is: ' + str(i))

    d = sio.loadmat('data/ex6data3.mat')
    X = d['X']
    y = d['y']
    Xval = d['Xval']
    yval = d['yval']

    (C, sigma, J) = dataset3params(X, y, Xval, yval)

    print ('C: ' + str(C))
    print ('sigma: ' + str(sigma))
    print ('Cost: ' + str(np.min(J)))

    print ('\nTesting the Gaussian Kernel\n')

    d = sio.loadmat('data/ex6data2.mat')
    X = d['X']
    y = d['y']

    plotData(X,y)

    clf = svm.SVC(kernel='rbf', gamma=50, C=1)
    clf.fit(X, y.ravel())
    visualBoundaryLinear(X, y, clf, 'Non Linear SVC')
