from classes.Kernels.Kernel import Kernel

class EpanechnikovKernel(Kernel):
    """
    Kernel Epanechnikov
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)     #super() -> pour appeler le parent
        self._name = "Kernel Epanechnikov"

    def kernelFunction(self, u):
        """
            Fonction à définir, associé au kernel. Voir page wikipédia sur les fonctions de kernel
        """
        abs_u = abs(u)
        if abs_u <= 1:
            return 3/4*(1-(u*u))
        return 0