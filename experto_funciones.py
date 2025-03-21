import numpy as np
from typing import Callable, Dict
import random

class ExpertoFunciones:
    def __init__(self):
        self.funciones: Dict[str, Callable] = {}
        self.operaciones_base = {
            'suma': lambda x, y: x + y,
            'multiplicacion': lambda x, y: x * y,
            'cuadrado': lambda x: x**2,
            'cubo': lambda x: x**3,
            'seno': lambda x: np.sin(x),
            'coseno': lambda x: np.cos(x)
        }
        self._inicializar_funciones_base()
    
    def _inicializar_funciones_base(self):
        """Inicializa funciones predeterminadas usando composición"""
        self.funciones = {
            'suma_cuadrados': lambda X: np.sum(X**2, axis=1),
            'suma_cubos': lambda X: np.sum(X**3, axis=1),
            'suma_senos': lambda X: np.sum(np.sin(X), axis=1),
            'multiplicacion_vars': lambda X: np.prod(X, axis=1),
            'mixta_nvars': lambda X: np.sum(X**2, axis=1) + np.prod(X, axis=1)
        }

    def agregar_funcion_compuesta(self, nombre: str, operaciones: list) -> None:
        """
        Agrega una función compuesta usando operaciones predefinidas
        operaciones: lista de tuplas (operacion, variable_idx, constante)
        """
        try:
            # Obtener solo las variables que realmente se usan
            vars_used = set(op[1] for op in operaciones)
            n_vars_needed = max(vars_used) + 1
            
            def nueva_funcion(X):
                if X.shape[1] > n_vars_needed:
                    X = X[:, :n_vars_needed]
                resultado = 0
                for op, var_idx, const in operaciones:
                    temp = X[:, var_idx]
                    # La constante se aplica antes de la operación
                    temp = temp if const == 1.0 else temp * const
                    if op in ['cuadrado', 'cubo', 'seno', 'coseno']:
                        temp = self.operaciones_base[op](temp)
                    resultado = self.operaciones_base['suma'](resultado, temp)
                return resultado
            
            self.funciones[nombre] = nueva_funcion
            vars_str = ", ".join(f"x{i+1}" for i in sorted(vars_used))
            print(f"Función '{nombre}' agregada exitosamente (usa variables: {vars_str})")
            
        except Exception as e:
            print(f"Error al agregar función: {str(e)}")

    def mostrar_funcion(self, operaciones: list) -> str:
        """Convierte las operaciones a una representación matemática legible"""
        # Generar nombres únicos para las variables
        var_names = {}  # var_idx -> nombre
        next_var = ord('x')  # Empezar con 'x'
        
        funcion_str = "f("
        # Construir lista de variables
        for op, var_idx, _ in operaciones:
            if var_idx not in var_names:
                var_name = chr(next_var)
                if var_idx > 0:  # Si hay más de una variable del mismo tipo
                    var_name += str(var_idx + 1)
                var_names[var_idx] = var_name
                next_var = min(ord('z'), next_var + 1)  # Siguiente letra, máximo 'z'
        
        # Mostrar variables usadas
        vars_list = [var_names[idx] for idx in sorted(var_names.keys())]
        funcion_str += ", ".join(vars_list) + ") = "
        
        # Construir expresión
        terms = []
        for op, var_idx, const in operaciones:
            term = ""
            var_name = var_names[var_idx]
            
            if op == 'cuadrado':
                term = f"{var_name}^2"
            elif op == 'cubo':
                term = f"{var_name}^3"
            elif op == 'multiplicacion':
                term = f"{var_name}"
            elif op == 'seno':
                term = f"sin({var_name})"
            elif op == 'coseno':
                term = f"cos({var_name})"
                
            if const != 1.0:
                term = f"{const}*{term}"
            terms.append(term)
        
        return funcion_str + " + ".join(terms)

    def generar_funcion_aleatoria(self) -> Callable:
        """Genera una función aleatoria usando operaciones seguras"""
        num_terminos = random.randint(2, 4)
        operaciones = []
        
        for _ in range(num_terminos):
            op = random.choice(['cuadrado', 'cubo', 'multiplicacion'])
            var_idx = random.randint(0, 1)
            const = random.uniform(-5, 5)
            operaciones.append((op, var_idx, const))
        
        nombre = f"aleatoria_{len(self.funciones)}"
        self.agregar_funcion_compuesta(nombre, operaciones)
        return self.funciones[nombre]

    def obtener_funcion(self, nombre: str) -> Callable:
        return self.funciones.get(nombre)

    def listar_funciones(self) -> None:
        print("\nFunciones disponibles:")
        for nombre in self.funciones:
            print(f"- {nombre}")

    def calcular_punto(self, x, operaciones):
        """Calcula el resultado de la función para un punto específico"""
        resultado = 0
        for op, var_idx, const in operaciones:
            valor = x[var_idx] * const
            if op in ['cuadrado', 'cubo', 'seno', 'coseno']:
                valor = self.operaciones_base[op](valor)
            resultado = self.operaciones_base['suma'](resultado, valor)
        return resultado

# Ejemplo de uso
if __name__ == "__main__":
    experto = ExpertoFunciones()
    
    # Agregar función compuesta
    operaciones = [
        ('cuadrado', 0, 2.0),  # 2*x^2
        ('cubo', 1, 1.0)       # + y^3
    ]
    experto.agregar_funcion_compuesta("mi_funcion", operaciones)
    
    # Generar y probar función aleatoria
    f_aleatoria = experto.generar_funcion_aleatoria()
    
    # Probar
    X_test = np.array([[1.0, 2.0]])
    print("\nPrueba de función:")
    f = experto.obtener_funcion("mi_funcion")
    if f is not None:
        print(f"f(1,2) = {f(X_test)[0]}")