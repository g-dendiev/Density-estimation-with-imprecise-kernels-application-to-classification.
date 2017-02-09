from classes.Kernels.Kernel import Kernel

class TriangularKernel(Kernel):
    """
    Kernel Triangulaire
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)                 #super() -> pour appeler le parent
        self._name = "Kernel Triangulaire"

    def kernelFunction(self, u):
        """
            Fonction à définir, associé au kernel.
        """
        abs_u = abs(u)
        if abs_u <= 1:
            return 1-abs_u
        return 0

