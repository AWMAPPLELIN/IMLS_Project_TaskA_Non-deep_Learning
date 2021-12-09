import os.path
import pickle as pk
from sklearn.decomposition import PCA


# The process to use PCA algorithm tp reduce dimension of the dataset and the interference of outliers
# x is served as a matrix with 3000 rows and 262144 columns
# n_components is the component of PCA algorithm
def pca_feature_extraction(x, n_components):
    pca_model_name = "pca_" + str(n_components) + ".pkl"
    if os.path.exists(pca_model_name):
        print("The process to load PCA model")
        pca = pk.load(open(pca_model_name, 'rb'))
    else:
        pca = PCA(n_components)
        pca.fit(x)
        pk.dump(pca, open(pca_model_name, "wb"))
    # The data after the operation of PCA algorithm
    pca_data = pca.transform(x)
    # The singular values, variance and estimated number of components of the selected components(n_components)
    singular_value = pca.singular_values_
    variance = pca.explained_variance_ratio_
    vcomp = pca.components_
    return pca_data, singular_value, variance, vcomp
