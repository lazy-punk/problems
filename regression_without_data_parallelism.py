
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from typing import Union
import sklearn.metrics as metrics
import jax
import optax

class LinearRegression:
    def __init__(self, learning_rate: float = 0.03, epochs: int = 10000, regularization_strength: float = 0.1,
                 data_regularization=True) -> None:
        """
        Initialize the LinearRegression object.

        Parameters:
        - learning_rate: Learning rate for gradient descent (default = 0.03).
        - epochs: Number of training iterations (default = 10000).
        - regularization_strength (default = 0.1)
        """
        self.n = None
        self.m = None
        try:
            self.cost = []
            self.epoch = []
            self.data_regularization = data_regularization
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.w = None
            self.b = None
            self.params = {}
        except Exception as e:
            print("An error occurred during initialization:", str(e))

    def Standard(self, X):
        SS = StandardScaler()
        g = SS.fit_transform(np.array(X, dtype=np.float32))
        return jnp.array(g)

    @partial(jax.jit, static_argnums=0)
    def forward(self, X, params):
        return jnp.dot(X, params['w']) + params['b']

    @partial(jax.jit, static_argnums=0)
    def loss(self, params, X, y):
        y_pred = self.forward(X, params)
        return jnp.mean(jnp.square(y_pred - y))

    @partial(jax.jit, static_argnums=0)
    def update(self, params, grads, opt_state):
        updates, opt_state = optax.adamw(self.learning_rate).update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], validation_split=0.2,
            early_stop_patience=5) -> None:
        """
        Train the linear regression model using gradient descent.
        Parameters:
        - X: Input features as a pandas DataFrame.
        - y: Target variable as a pandas Series.
        """
        try:
            X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
            if self.data_regularization:
                X = self.Standard(X)
            self.m, self.n = X.shape
            self.w = jnp.array(np.random.normal(size=self.n) * 1e-4)
            self.b = jnp.array(0.0)
            self.params = {'w': self.w, 'b': self.b}
            description = tqdm(range(self.epochs))
            X, X_test, y, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)
            best_val_loss = float('inf')
            best_val_acc = float('-inf')
            patience = early_stop_patience
            solver = optax.adamw(learning_rate=0.003)
            opt_state = solver.init(self.params)
            if metrics.r2_score(y, self.forward(X, self.params)) < 0:
                tqdm.write("Negative R2Score, wait for a while")
            for i in description:

                acc = 0
                loss = 0
                description.set_description(f"R2Score:{metrics.r2_score(y, self.forward(X, self.params))}")
                for _ in range(10):
                    loss, grads = jax.value_and_grad(self.loss, argnums=0, allow_int=True)(self.params, X, y)
                    acc = round(metrics.r2_score(y, self.forward(X, self.params)), 5)
                    self.params, opt_state = self.update(self.params, grads, opt_state)
                    # self.params = self.update(self.params, grads)
                    self.cost.append(loss)
                    self.epoch.append(i)

                if acc <= best_val_acc or loss >= best_val_loss:
                    patience -= 1
                else:
                    best_val_loss = loss
                    best_val_acc = acc
                    patience = early_stop_patience

                if patience == 0:
                    tqdm.write(f"Stopping early at epoch {i+1} due to constant or slow convergence rate")
                    if metrics.r2_score(y, self.forward(X, self.params)) < .5:
                        print('Try changing the hyperparameters')
                    description.close()
                    break
            if metrics.r2_score(y, self.forward(X, self.params)) <= .5:
                print("Model isn't working well try: ")
                print("1. Changing the Hyperparameters")
                print("2. Changing the Model e.x., MLPRegressor")

        except Exception as e:
            print("An error occurred during fitting:", str(e))

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable for the given input features.

        Parameters:
        - X_test: Input features for prediction as a pandas DataFrame.

        Returns:
        - Predicted target variable as a numpy array.
        """
        if self.data_regularization:
            X_test = self.Standard(X_test)
        return self.forward(X_test, self.params)

    def plot_cost(self) -> None:
        """
        Plot the cost function over training iterations.
        """
        plt.plot(self.cost, self.epoch)
        plt.show()

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Evaluate the model using the R-squared metric.

        Parameters:
        - X_test: Test input features as a numpy array.
        - y_test: Test target variable as a numpy array.
        """
        return(metrics.r2_score(y_true, y_pred))
    
from sklearn.datasets import _samples_generator

x, y = _samples_generator.make_regression(n_samples=5000, n_features=30, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression(learning_rate=0.03, epochs=10000, regularization_strength=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.evaluate(y_test, y_pred))
