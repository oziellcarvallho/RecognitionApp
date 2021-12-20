from cv2 import ml, TERM_CRITERIA_MAX_ITER
from kivy.utils import platform
from kivy.logger import Logger
from pickle import load
from numpy import mean
import os

def accuracy(y_true, y_pred, normalize=True):
    accuracy=[]
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    if normalize==True:
        return mean(accuracy)
    if normalize==False:
        return sum(accuracy)

def train(samples, flags, x_teste):
    classifier = ml.RTrees_create()
    
    classifier.setMaxDepth(17)
    classifier.setMaxCategories(6)
    classifier.setCalculateVarImportance(True)
    classifier.setMinSampleCount(0)
    
    term_type, n_trees, forest_accuracy = TERM_CRITERIA_MAX_ITER, 300, 1
    classifier.setTermCriteria((term_type, n_trees, forest_accuracy))
    
    train_data = ml.TrainData_create(samples=samples, layout=ml.ROW_SAMPLE, responses=flags)
    classifier.train(train_data)
    
    if platform == 'android':
        from android.storage import primary_external_storage_path
        dir = primary_external_storage_path()
        download_dir_path = os.path.join(dir, 'Download')
        classifier.save(download_dir_path + '/MA.sav')
    
    _ret, responses = classifier.predict(x_teste)
    return responses.ravel()

def pickle_load(filename):
    data = []
    try:
        with open(filename,'rb') as infile:
            data = load(infile)
            infile.close()
    except:
        Logger.exception('Recognition: Erro Open Pickle')
    return data

def load_data():
    path_save = './data/model/'
    
    x_treino = pickle_load('./data/model/Data_Treino.pickle')
    y_treino = pickle_load('./data/model/Label_Treino.pickle')
    x_teste = pickle_load('./data/model/Data_Teste.pickle')
    y_teste = pickle_load('./data/model/Label_Teste.pickle')

    try:
        pred = train(x_treino, y_treino, x_teste)
        acc = accuracy(y_teste, pred)
    except:
        acc = None
    return str(acc)