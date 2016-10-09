from classes.Kernels.Kernel import Kernel

class TriangularKernel(Kernel):
    """
    Kernel Triangulaire
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)     #super() -> pour appeler le parent
        self._name = "Kernel Triangulaire"

    def value(self, centerPoint, seekedPoint):
        if abs(centerPoint - seekedPoint) <= (self.bandwidth / 2):
            # Thales !!!!
            return 2/self._bandwidth - 4*abs(centerPoint - seekedPoint)/(self._bandwidth*self.bandwidth)
        else:
            # Pas dans le Kernel
            return 0