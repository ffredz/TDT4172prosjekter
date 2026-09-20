import numpy as np
from sklearn.base import BaseEstimator


class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=10000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        self.pred_err = []

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        X=np.asarray(X).flatten()
        y=np.asarray(y).flatten()
        self.weights = 0
        self.bias = 0
        n = len(X)
        
        for _ in range(self.n_iterations):
            MSE = np.mean((y-(self.weights*X+self.bias))**2)
       
            (self.loss_history).append(MSE)

            grad_w = -2*np.mean(X*(y-(self.weights*X+self.bias)))
            grad_b = -2*np.mean(y-(self.weights*X+self.bias))
            self.bias-=self.lr*grad_b
            self.weights-=self.lr*grad_w 
        for k in range(n):
            residual = y[k]-(self.weights*X[k]+self.bias)
            self.pred_err.append(residual)
        
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        X=np.asarray(X).flatten()
        return self.weights*X+self.bias 
        

class LogisticRegression(BaseEstimator):
    def __init__(self, lr=0.1, n_iterations=10000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        self.accuracies = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        self.weights = np.zeros(X.shape[1])
        self.bias = 0 

        for _ in range(self.n_iterations):
            lin_model = np.dot(X, self.weights) + self.bias 
            y_pred = self.sigmoid(lin_model)
            grad_w = np.dot(X.T, (y_pred-y)) / len(y_pred)
            grad_b = np.mean(y_pred-y)
            self.weights = self.weights- self.lr*grad_w 
            self.bias = self.bias - self.lr*grad_b 
            loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
            self.loss_history.append(loss)
            (self.accuracies).append(self.accuracy(y, (y_pred >= 0.5).astype(int)))
        
            
       
    def accuracy(self, y,y_pred):
        return np.mean(y==y_pred)
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        linear_model = np.dot(X, self.weights) + self.bias 
        return self.sigmoid(linear_model)
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        return (self.predict_proba(X)>=.5).astype(int)
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        return 1/(1+np.exp(-z))
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")