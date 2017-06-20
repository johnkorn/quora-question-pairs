import pickle

data_mapper = {data_1: 'traindata1', data_2: 'traindata2',
               test_data_1: 'testdata1', test_data_2: 'testdata2',
               labels: 'trainlabels', test_ids: 'testids',
               embedding_matrix: 'w2vmatrix'}

droot = 'DATA/processed/'
ext = '.pkl'

def load_data(dname):
    fname = droot+data_mapper[dname]+ext
    return pickle.load(open(fname, 'rb'))