import numpy as np

class VistaConsola:
    def __init__(self):
        self.controlador = None

    def set_controlador(self, controlador):
        self.controlador = controlador

    def mostrar_menu(self):
        while True:
            print("\n=== MENÚ RBF ===")
            print("1. Ejecutar pruebas")
            print("2. Mostrar funciones disponibles")
            print("3. Crear nueva función")
            print("4. Salir")
            
            opcion = input("Seleccione una opción: ")
            if opcion == "1":
                self.solicitar_pruebas()
            elif opcion == "2":
                self.mostrar_funciones()
            elif opcion == "3":
                self.solicitar_nueva_funcion()
            elif opcion == "4":
                break

    def generar_cabecera_tabla(self, n_vars):
        header = ""
        for i in range(n_vars):
            header += f"{'X'+str(i+1):^8}"
        header += f"{'Real':^12}{'Pseudo':^12}{'Gradient':^12}{'Error P':^10}{'Error G':^10}"
        return header

    def print_resultados(self, pred_pseudo, pred_gradient, valores_reales, X_test, operaciones):
        n_vars = X_test.shape[1]
        print("\n" + "="*70)
        print("COMPARACIÓN DE MÉTODOS DE ENTRENAMIENTO")
        print("="*70)
        
        print(f"Función objetivo: {self.controlador.mostrar_funcion(operaciones)}")
        print(f"Número de variables de entrada: {n_vars}")
        print("-"*70)
        
        print(self.generar_cabecera_tabla(n_vars))
        print("-"*70)
        
        error_pseudo = []
        error_gradient = []
        
        for i in range(len(X_test)):
            row = ""
            for j in range(n_vars):
                row += f"{X_test[i,j]:8.2f}"
            
            error_p = abs(valores_reales[i]-pred_pseudo[i])
            error_g = abs(valores_reales[i]-pred_gradient[i])
            error_pseudo.append(error_p)
            error_gradient.append(error_g)
            
            print(f"{row}{valores_reales[i]:12.3f}{pred_pseudo[i]:12.3f}"
                  f"{pred_gradient[i]:12.3f}{error_p:10.3f}{error_g:10.3f}")
        
        print("-"*70)
        print(f"Error medio absoluto (Pseudoinversa): {np.mean(error_pseudo):.6f}")
        print(f"Error medio absoluto (Gradiente): {np.mean(error_gradient):.6f}")
        print(f"Error máximo (Pseudoinversa): {max(error_pseudo):.6f}")
        print(f"Error máximo (Gradiente): {max(error_gradient):.6f}")
        print("="*70)

        # Añadir información detallada de la red
        print("\nDETALLES DE LA RED RBF")
        print("="*70)
        print(f"Número de neuronas RBF: {self.controlador.red.centers.shape[0]}")
        print(f"Dimensión de entrada: {self.controlador.red.centers.shape[1]}")
        print(f"Valor de sigma: {self.controlador.red.sigma:.4f}")
        
        print("\nCENTROS DE LAS RBF:")
        for i, center in enumerate(self.controlador.red.centers):
            print(f"Neurona {i+1}: {center}")
        
        print("\nPESOS FINALES:")
        print(f"Pseudoinversa: {self.controlador.red.weights}")
        
        print("\nDATOS DE ENTRENAMIENTO:")
        print(f"Número de puntos: {len(X_test)}")
        print(f"Rango de entrada: [{X_test.min():.2f}, {X_test.max():.2f}]")
        
        print("\nESTADÍSTICAS DE ERROR")
        # Evitar división por cero en error relativo
        error_pseudo_rel = []
        error_gradient_rel = []
        for real, pseudo, grad in zip(valores_reales, pred_pseudo, pred_gradient):
            if abs(real) > 1e-10:  # Umbral para considerar no-cero
                error_pseudo_rel.append(abs(pseudo - real)/abs(real))
                error_gradient_rel.append(abs(grad - real)/abs(real))
            else:
                # Para valores cercanos a cero, usar error absoluto
                error_pseudo_rel.append(abs(pseudo - real))
                error_gradient_rel.append(abs(grad - real))
        
        error_pseudo_rel = np.mean(error_pseudo_rel) * 100
        error_gradient_rel = np.mean(error_gradient_rel) * 100
        print(f"Error relativo medio (Pseudoinversa): {error_pseudo_rel:.2f}%")
        print(f"Error relativo medio (Gradiente): {error_gradient_rel:.2f}%")
        print("="*70)

    def solicitar_pruebas(self):
        n_vars = 4
        operaciones = [
            ('cuadrado', 0, 1.0),      # x1^2
            ('cubo', 1, 1.0),          # + x2^3
            ('multiplicacion', 2, 1.0), # + x3
            ('seno', 3, 1.0)           # + sin(x4)
        ]
        
        print("Funciones disponibles antes de agregar nueva función:")
        self.mostrar_funciones()
        
        X_test, pred_pseudo, pred_gradient, valores_reales = self.controlador.ejecutar_pruebas(n_vars, operaciones)
        
        if X_test is None:
            print("Error al crear la red")
            return
            
        self.controlador.mostrar_historial_entrenamiento()

    def mostrar_funciones(self):
        funciones = self.controlador.obtener_funciones_disponibles()
        for funcion in funciones:
            print(f"- {funcion}")

    def solicitar_nueva_funcion(self):
        nombre = input("Nombre de la nueva función: ")
        n_operaciones = int(input("Número de operaciones: "))
        n_neurons = int(input("Número de neuronas RBF (recomendado 3-10): "))
        operaciones = []
        vars_used = set()
        
        print("\nOperaciones disponibles:")
        ops = {
            '1': 'cuadrado',
            '2': 'cubo',
            '3': 'multiplicacion',
            '4': 'seno',
            '5': 'coseno'
        }
        
        var_names = {}  # Mapeo de índices a nombres de variables
        next_var = ord('x')  # Empezar con 'x'
        
        for i in range(n_operaciones):
            print("\nOperación", i+1)
            print("1. Cuadrado (x^2)")
            print("2. Cubo (x^3)")
            print("3. Multiplicación (x)")
            print("4. Seno (sin(x))")
            print("5. Coseno (cos(x))")
            
            while True:
                try:
                    op_choice = input("Seleccione operación (1-5): ")
                    if op_choice not in ops:
                        print("Operación no válida. Intente de nuevo.")
                        continue
                    
                    print("\nVariables disponibles:")
                    for idx in sorted(vars_used):
                        var_name = chr(next_var) if idx == 0 else f"{chr(next_var)}{idx+1}"
                        print(f"  {idx}: {var_name}")
                    print(f"  {len(vars_used)}: Nueva variable")
                    
                    var_idx = int(input("Seleccione variable: "))
                    if var_idx < 0:
                        print("El índice debe ser positivo")
                        continue
                    
                    vars_used.add(var_idx)
                    const = float(input("Constante multiplicativa (por defecto 1.0): ") or "1.0")
                    operaciones.append((ops[op_choice], var_idx, const))
                    break
                except ValueError as e:
                    print("Error: Entrada no válida. Intente de nuevo.")
        
        n_vars = max(vars_used) + 1
        
        self.controlador.crear_funcion(nombre, operaciones)
        print(f"\nFunción creada: {self.controlador.mostrar_funcion(operaciones)}")
        
        X_test, pred_pseudo, pred_gradient, valores_reales = self.controlador.ejecutar_pruebas(n_vars, operaciones, n_neurons)
        
        if X_test is not None:
            self.print_resultados(pred_pseudo, pred_gradient, valores_reales, X_test, operaciones)
            self.controlador.mostrar_historial_entrenamiento()
