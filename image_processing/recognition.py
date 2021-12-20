from cv2 import ml, xfeatures2d, MSER_create, imread
from numpy import array

file_name = './data/model/MA.sav'

mser = MSER_create(_delta = 2)
surf = xfeatures2d.SURF_create(hessianThreshold = 200, nOctaves = 22, nOctaveLayers = 21)
loaded_model = ml.RTrees_load(file_name)

def classify(image):
    keys = mser.detect(image, None)
    kp, des = surf.compute(image, keys)
    dado = []
    a = []
    for j in range(64):
        a.append(des.transpose()[j].mean())
    dado.append(a)
    dados = array(dado)
    responses = loaded_model.predict(dados)
    return str(responses[0])[:-2] + ' Reais'