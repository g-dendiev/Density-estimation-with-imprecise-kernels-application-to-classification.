class Kernel:
    """
    Super classe générique des Kernels,
    sert de modèle & permet le polymorphisme.
    """

    # Constructeur de classe
    def __init__(self, bandwidth):
        self._name = "Undefined"
        self._bandwidth = bandwidth

    # Getter / Setter de Bandwidth -> domaine sur lequel on fait passer le Kernel
    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth

    # Getter / Setter du nom
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        print("On ne change pas le nom du Kernel !")
