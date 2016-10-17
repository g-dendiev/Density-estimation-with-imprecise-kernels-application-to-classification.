from classes.Kernels.Kernel import Kernel

from math import pi
from math import sin
from math import acos #arcos



class EllipseKernel(Kernel):
    """
    Kernel en demi-cercle initialement, puis hMin ethMax provoque un changement en ellipse
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)     #super() -> pour appeler le parent
        self._name = "Kernel Ellipsoidal"

    def value(self, centerPoint, seekedPoint):
        if abs(centerPoint - seekedPoint) <= (self.bandwidth / 2):
            # Cos et sin pour le demi-cercle ! Pour l'ellipse : passage par : x=a*cos(t) et y=b*sin(t)
            return (4/(pi*self.bandwidth))*sin(acos((2*(centerPoint - seekedPoint)/self.bandwidth)))
        else:
            # Pas dans le Kernel
            return 0