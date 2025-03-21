from controlador import Controlador
from Vista_Consola import VistaConsola

def main():
    controlador = Controlador()
    vista = VistaConsola()

    controlador.set_vista(vista)
    vista.mostrar_menu()

if __name__ == "__main__":
    main()
