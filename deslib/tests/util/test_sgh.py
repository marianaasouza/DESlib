import pytest

from deslib.util.sgh import _build_Perceptron, SGH
from deslib.tests.examples_test import *

@pytest.mark.parametrize('incl_samples, expected', [(np.ones(15,dtype=int), [[[-0.25,0.05555556],[0.66666667,0.25]],[[-0.97823413,-0.20750421],[0.97823413,0.20750421]],[0.23550080789503214,-0.23550080789503214]]),
                                             (np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],dtype=int), [[[-0.25,0.0],[0.58333333,0.5]],[[-0.85749293,-0.51449576],[0.85749293,0.51449576]],[0.271539426475639,-0.271539426475639]]),
                                             (np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],dtype=int), [[[-0.25,-0.3],[0.75,0.0]],[[-0.95782629,-0.28734789],[0.95782629,0.28734789]],[0.19635438847033604,-0.19635438847033604]])])

def test_build_Perceptron(incl_samples, expected):
    n_classes = 2
    n_features = 2
    init_centroids = np.zeros((n_classes,n_features),float)
    classif_params = _build_Perceptron(X_dsel_ex1,y_dsel_ex1,incl_samples,init_centroids)

    assert all([np.allclose(a, b) for a,b in zip(expected,classif_params)])

def test_generate_pool():
    # Coefficients and intercepts of resulting classifiers in the SGH pool
    expected = [[np.array([[-0.97823413,-0.20750421],[0.97823413,0.20750421]]),np.array([0.23550080789503214,-0.23550080789503214])],
                [np.array([[1.0,0.0],[ -1.0,0.0]]),np.array([-0.41666667, 0.41666667])]]

    sgh_ex = SGH()
    
    sgh_ex.fit(X_dsel_ex1,y_dsel_ex1)

    pool_params = [[c.coef_,c.intercept_] for c in sgh_ex.estimators_]

    assert all([np.allclose(c, d) for a,b in zip(expected,pool_params) for c,d in zip(a,b)])

def test_same_class_centroids():
    # Coefficients and intercepts of resulting classifiers in the SGH pool
    expected = [[np.array([[ 0.70710678,  0.70710678],[-0.70710678, -0.70710678]]),np.array([ 0.0, 0.0])],[np.array([[-0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]]),np.array([ 0.35355339, -0.35355339])]]

    X_ex_concentric = np.array([[1,0],[-1,0],[0,1],[0,-1],[2,0],[-2,0],[0,2],[0,-2]])
    y_ex_concentric = np.array([0,0,0,0,1,1,1,1])

    sgh_ex = SGH()
    
    sgh_ex.fit(X_ex_concentric,y_ex_concentric)

    pool_params = [[c.coef_,c.intercept_] for c in sgh_ex.estimators_]

    assert all([np.allclose(c, d) for a,b in zip(expected,pool_params) for c,d in zip(a,b)])



