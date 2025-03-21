from RBF import RBF
import numpy as np
from experto_funciones import ExpertoFunciones

class Controlador:
    def __init__(self):
        self.vista = None
        self.experto_funciones = ExpertoFunciones()
        self.red = None

    def set_vista(self, vista):
        self.vista = vista
        vista.set_controlador(self)

    def obtener_funciones_disponibles(self):
        return list(self.experto_funciones.funciones.keys())

    def crear_funcion(self, nombre, operaciones):
        self.experto_funciones.agregar_funcion_compuesta(nombre, operaciones)

    def mostrar_funcion(self, operaciones):
        """Retorna representación de la función"""
        return self.experto_funciones.mostrar_funcion(operaciones)

    def ejecutar_pruebas(self, n_vars, operaciones, n_neurons=5):
        # Validar número mínimo de neuronas
        n_neurons = max(1, n_neurons)  # Asegurar al menos 1 neurona
        
        vars_used = set(op[1] for op in operaciones)
        n_vars_needed = max(vars_used) + 1
        
        if n_vars != n_vars_needed:
            n_vars = n_vars_needed

        self.experto_funciones.agregar_funcion_compuesta("funcion1", operaciones)
        funcion1 = self.experto_funciones.obtener_funcion("funcion1")
        if funcion1 is None:
            return None, None, None, None

        X_test = np.array([np.linspace(-2, 2, 5) for _ in range(n_vars)]).T
        self.red = RBF(n_neurons=n_neurons, input_dim=n_vars, 
                      target_function=funcion1, sigma=None)

        self.red.calcular_pseudoInversa(X_test)
        pred_pseudo = self.red.predict(X_test)

        self.red.train_gradient_descent(X_test, learning_rate=0.01, epochs=1000)
        pred_gradient = self.red.predict(X_test)

        valores_reales = funcion1(X_test)
        return X_test, pred_pseudo, pred_gradient, valores_reales

    def mostrar_historial_entrenamiento(self):
        if self.red:
            self.red.plot_training_history()
