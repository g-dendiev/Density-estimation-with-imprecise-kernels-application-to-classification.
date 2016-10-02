from Kernel import Kernel

class TriangularKernel(Kernel):
    """
    Kernel Triangulaire
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)
        self._name = "Kernel Triangulaire"


a = TriangularKernel(1)
print(a.name)
print(a.bandwidth)