import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf_matrix(X, centers, sigma):
    """Calcula la matriz de activaciones RBF manteniendo independencia dimensional"""
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    n_dims = X.shape[1]
    
    # verificaciones:
    assert n_samples > 0, "No hay muestras de entrada"
    assert n_centers > 0, "No hay centros definidos"
    
    # Calcular distancias por dimensión
    phi = np.ones((n_samples, n_centers))
    for d in range(n_dims):
        x_d = X[:, d:d+1]  # Mantener dimensión
        c_d = centers[:, d:d+1].T  # Transponer para broadcasting
        dist_d = (x_d - c_d)**2
        phi *= np.exp(-dist_d / (2 * sigma**2))
    
    return phi


class RBF:
    def __init__(self, n_neurons, input_dim, target_function, sigma=1.0):
        """Inicializa la red RBF"""
        assert n_neurons > 0 and input_dim > 0
        
        # Distribuir centros estratégicamente en el espacio de entrada
        self.centers = np.zeros((n_neurons, input_dim))
        if n_neurons == 1:
            self.centers[0] = np.zeros(input_dim)  # Centro en el origen
        else:
            # Distribuir centros uniformemente en una cuadrícula
            points = np.linspace(-2, 2, int(np.ceil(np.sqrt(n_neurons))))
            grid = np.array(np.meshgrid(*[points for _ in range(input_dim)]))
            grid_points = grid.reshape(input_dim, -1).T
            
            # Seleccionar n_neurons puntos más cercanos al origen
            distances = np.linalg.norm(grid_points, axis=1)
            idx = np.argsort(distances)[:n_neurons]
            self.centers = grid_points[idx]
        
        # Ajustar sigma basado en la separación entre centros
        if sigma is None:
            if n_neurons == 1:
                self.sigma = 2.0  # Cubrir todo el rango de entrada
            else:
                # Calcular distancias entre todos los pares de centros usando numpy
                dists = []
                for i in range(n_neurons):
                    for j in range(i + 1, n_neurons):
                        dist = np.linalg.norm(self.centers[i] - self.centers[j])
                        dists.append(dist)
                self.sigma = np.mean(dists) / np.sqrt(2*n_neurons)
        else:
            self.sigma = sigma

        self.target_function = target_function
        self.weights = None
        self.training_history = {
            'loss': [],
            'epochs': [],
            'weights': [],
            'learning_rate': None
        }
        self.loss_history = []  # Add this line

    def calcular_pseudoInversa(self, X_train):
        """Entrena la red usando la pseudoinversa"""
        phi = gaussian_rbf_matrix(X_train, self.centers, self.sigma)
        y_train = self.target_function(X_train)
        
        # Usar SVD con regularización
        U, S, Vh = np.linalg.svd(phi, full_matrices=False)
        # Regularización más robusta
        lambda_reg = 1e-6
        S_inv = S / (S**2 + lambda_reg)
        self.weights = (Vh.T * S_inv) @ U.T @ y_train
        
        # Verificar predicción inmediata
        y_pred = phi @ self.weights
        error = np.mean(np.abs(y_train - y_pred))
        print(f"Error de entrenamiento (pseudoinversa): {error:.6f}")
        
        return self.weights
    
    def train_gradient_descent(self, X_train, learning_rate=0.01, epochs=1000):
        """Entrena la red usando descenso de gradiente y guarda el historial"""
        phi = gaussian_rbf_matrix(X_train, self.centers, self.sigma)
        y_train = self.target_function(X_train)
        
        # Inicializar pesos con pequeños valores aleatorios centrados en cero
        self.weights = np.random.normal(0, 0.01, size=self.centers.shape[0])
        
        # Limpiar historiales
        self.loss_history = []
        self.training_history['loss'] = []
        self.training_history['epochs'] = []
        self.training_history['weights'] = []
        self.training_history['learning_rate'] = learning_rate
        
        best_loss = float('inf')
        best_weights = None
        patience = 50  # épocas sin mejora antes de reducir learning rate
        no_improve_count = 0
        min_lr = 1e-6
        
        for epoch in range(epochs):
            y_pred = phi @ self.weights
            error = y_train - y_pred
            loss = np.mean(error**2)
            
            # Guardar historiales
            self.loss_history.append(loss)
            self.training_history['loss'].append(loss)
            self.training_history['epochs'].append(epoch)
            self.training_history['weights'].append(self.weights.copy())
            
            # Verificar mejora
            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Reducir learning rate si no hay mejora
            if no_improve_count >= patience and learning_rate > min_lr:
                learning_rate *= 0.5
                no_improve_count = 0
                print(f"Reduciendo learning rate a {learning_rate}")
            
            # Early stopping
            if learning_rate < min_lr:
                print("Learning rate muy pequeño, deteniendo entrenamiento")
                break
                
            gradient = -2 * phi.T @ error / len(X_train)
            self.weights -= learning_rate * gradient
            
            if epoch % 100 == 0:
                print(f"Época {epoch}, Loss: {loss:.6f}, LR: {learning_rate:.6f}")
        
        # Usar los mejores pesos encontrados
        self.weights = best_weights
        return self.weights
        
    def predict(self, X):
        """Realiza predicciones para nuevos datos"""
        if self.weights is None:
            raise ValueError("La red debe ser entrenada primero")
        phi = gaussian_rbf_matrix(X, self.centers, self.sigma)
        return phi @ self.weights
    
    def get_error(self, X_test):
        """Calcula el error medio absoluto"""
        y_pred = self.predict(X_test)
        y_true = self.target_function(X_test)
        return np.mean(np.abs(y_pred - y_true))
    
    def plot_training_history(self):
        """Visualiza el historial de entrenamiento"""
        if not self.training_history['loss']:
            print("No hay historial de entrenamiento disponible")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['epochs'], self.training_history['loss'])
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Época')
        plt.ylabel('Error Cuadrático Medio')
        plt.grid(True)
        plt.show()




