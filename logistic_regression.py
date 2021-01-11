import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self):
        self._coeff = []
    
    def fit(self, X, Y, num_iterations=50000, learning_rate=0.001, lmbda=0, plot_cost=True):
        _, params, _ = self._gradient_descent(X, Y, num_iterations, learning_rate, lmbda, plot_cost)
        self._coeff = params

    def predict(self, X, decision_boundary=0.5):
        X = np.vstack([np.ones(X.shape[0]), X.T])
        y_hat = self._sigmoid(np.dot(X.T, self._coeff))
        return np.where(y_hat > decision_boundary, 1, 0)

    def score(self, y_true, y_pred):
        assert len(y_true) == len(y_pred), "Length must be equal"
        accuracy = 100*np.squeeze(sum(np.where(y_true == y_pred,1, 0))/len(y_true))
        print(f'{round(accuracy, 5)} % accuracy')

    def _gradient_descent(self, X, Y, num_iterations, learning_rate, lmbda, plot_cost):
            X = np.vstack([np.ones(X.shape[0]), X.T])           
            theta = np.zeros(X.shape[0]).reshape(-1, 1)         # initializing weights and bias
            m = len(Y)
            costs = []
            
            y_hat = self._sigmoid(np.dot(X.T, theta))
            
            for i in range(num_iterations):
                theta = theta - learning_rate*((1/m)*np.dot(X, (y_hat-Y)) + (lmbda/m)*theta)
                y_hat = self._sigmoid(np.dot(X.T, theta))
                cost = self._calculate_loss(Y, y_hat, theta, lmbda)
                
                if plot_cost and i % 1000 == 0:
                    print(f'The cost at iteration {i} is {cost}')
                if plot_cost and i % 10 == 0:
                    costs.append(cost)
            
            if plot_cost:
                plt.plot(costs)
                plt.xlabel('Number of iterations (per tenth)')
                plt.ylabel('Cost')
                plt.title('Learning rate = ' + str(learning_rate))
            
            return y_hat, theta, cost

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _calculate_loss(self, Y, y_hat, theta, lmbda=0):
        """Calculates the binary cross entropy loss"""
        m = len(Y)
        loss = (-1/m)*np.sum((Y*np.log(y_hat)) + (1-Y)*np.log(1-y_hat))

        if lmbda:
            loss = loss + np.squeeze((lmbda/2/m)*np.dot(theta.T, theta))
        
        return loss