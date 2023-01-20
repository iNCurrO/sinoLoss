import numpy as np

def GeoCal(CTGeo):
    print("Generation Amatrix. Given Geometry of CT system is: \n")
    Amatrix = None
    print("Amatrix generation done...")
    return Amatrix


def rotation(x, y, theta):
    return [
        x*np.cos(theta) - y*np.sin(theta),
        x*np.sin(theta) + y*np.cos(theta)
    ]
