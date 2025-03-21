import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf_matrix(X, centers, sigma):
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    # verificaciones:
    assert n_samples > 0, "No hay muestras de entrada"
    assert n_centers > 0, "No hay centros definidos"
    # Reshape para hacer broadcasting
    X_expanded = X[:, np.newaxis]  # Shape: (n_samples, 1, n_features)
    centers_expanded = centers[np.newaxis, :]  # Shape: (1, n_centers, n_features)
    # Calcular distancias entre todos los puntos
    distances = np.sum((X_expanded - centers_expanded) ** 2, axis=2)
    # Aplicar función gaussiana
    return np.exp(-distances / (2 * sigma**2))


class RBF:
    def __init__(self, n_neurons, input_dim, target_function, sigma=1.0):
        """Inicializa la red RBF"""
        assert n_neurons > 0 and input_dim > 0
        
        # Distribuir más centros en el rango para mejor aproximación
        self.centers = np.array([
            np.linspace(-3, 3, n_neurons) for _ in range(input_dim)
        ]).T
        
        # Ajustar sigma más fino para funciones con constantes
        if sigma is None:
            if n_neurons == 1:
                # Para una sola neurona, usar un sigma predeterminado
                self.sigma = 1.0
            else:
                d = np.mean([
                    np.min([np.linalg.norm(c1-c2) 
                           for j, c2 in enumerate(self.centers) if i != j])
                    for i, c1 in enumerate(self.centers)
                ])
                self.sigma = d / np.sqrt(n_neurons)  # Ajuste más suave
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
        
        # Limpiar historiales
        self.loss_history = []
        self.training_history['loss'] = []
        self.training_history['epochs'] = []
        self.training_history['weights'] = []
        self.training_history['learning_rate'] = learning_rate
        
        # Inicializar pesos aleatoriamente
        self.weights = np.random.randn(self.centers.shape[0]) * 0.1  # Menor magnitud inicial
        
        best_loss = float('inf')
        best_weights = None
        
        for epoch in range(epochs):
            y_pred = phi @ self.weights
            error = y_train - y_pred
            loss = np.mean(error**2)
            
            # Guardar historiales
            self.loss_history.append(loss)
            self.training_history['loss'].append(loss)
            self.training_history['epochs'].append(epoch)
            self.training_history['weights'].append(self.weights.copy())
            
            # Guardar los mejores pesos
            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()
            
            gradient = -2 * phi.T @ error / len(X_train)
            self.weights -= learning_rate * gradient
            
            if epoch % 100 == 0:
                print(f"Época {epoch}, Loss: {loss:.6f}")
        
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




