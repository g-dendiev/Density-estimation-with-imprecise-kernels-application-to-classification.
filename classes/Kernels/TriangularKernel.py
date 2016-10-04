from classes.Kernels.Kernel import Kernel

class TriangularKernel(Kernel):
    """
    Kernel Triangulaire
    """

    # Constructeur de classe
    def __init__(self, h = 1):
        super().__init__(h)
        self._name = "Kernel Triangulaire"

    def value(self, centerPoint, seekedPoint):
        if abs(centerPoint - seekedPoint) < (self.bandwidth / 2):
            # Thales !!!!
            return 4 / (self.bandwidth*self.bandwidth) * abs(centerPoint - seekedPoint)
        else:
            # Pas dans le Kernel
            return 0