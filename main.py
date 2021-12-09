import os
import numpy as np
import pickle as pk
import Data_Pre_Processing
import PCA_Feature_Extraction
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# The path of dataset - image
data_direction = "dataset-0/image"
# The path of dataset - label
label_path = "dataset-0/label.csv"
# The path of a extreme large matrix that saves images and labels
original_mri_matrix_path = "MRI_Image_Matrix.npy"

# SVM with PCA method for binary classification


def svm_with_pca(x, y, n_components):
    # x is served as a numpy matrix with 3000 rows and 262144 columns
    # y is the label matrix with 0 to be no tumor and 1 to be tumor
    # k is the component of PCA algorithm
    # The process to split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    # The process to show x_train and its shape before PCA_Feature_Extraction
    print("The dataset x_train before PCA:\n", x_train.shape)
    print(x_train)
    # The process to show y_train and its shape before PCA_Feature_Extraction
    print("The dataset y_train before PCA:\n", y_train.shape[0])
    print(y_train)
    # The process to do the PCA operation
    x_train, singular, variance, vcomp = PCA_Feature_Extraction.pca_feature_extraction(x_train, n_components)
    print(singular)
    print(variance)
    print(vcomp)
    x_train = np.array(x_train)
    # The process to show x_train and its shape after PCA_Feature_Extraction
    print("The dataset x_train after PCA:\n", x_train.shape)
    print(x_train)
    # The process to show y_train and its shape after PCA_Feature_Extraction
    print("The dataset y_train after PCA:\n", y_train.shape[0])
    print(y_train)

    # PCA Process
    # The process to load PCA model from the file
    pca_model_name = "PCA_" + str(n_components) + ".pkl"
    trained_pca_model = pk.load(open(pca_model_name, 'rb'))
    # The process to do transform operation of x_test
    x_test = trained_pca_model.transform(x_test)
    # The process to show y_test and its shape after PCA_Feature_Extraction
    print("x_test after PCA:\n", x_test.shape)
    print(x_test)

    # SVM Process
    # The process to search the best parameter with the function parameter_grid and GridSearchCV
    svc_model = svm.SVC(kernel='poly')
    parameter_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    svc_model_grid_search = GridSearchCV(svc_model, parameter_grid, n_jobs=8, verbose=1)
    # The process to train SVC model
    svc_model_grid_search.fit(x_train, y_train.ravel())
    # The process to test SVC model
    y_prediction = svc_model_grid_search.predict(x_test)
    # The process to show the combination of parameters that have achieved the best result
    print("The best parameters are" % svc_model_grid_search.best_params_, svc_model_grid_search.best_params_)
    # The process to show the best score observed during the optimization process
    print("The best scores are" % svc_model_grid_search.best_params_, svc_model_grid_search.best_score_)
    print("The predicted data is:")
    print(np.array(y_prediction))
    print("The actual data is:")
    print(np.array(y_test))
    score = accuracy_score(y_prediction, y_test)
    # The process to obtain the accuracy
    print(f"The proposed svm_with_pca model has an accuracy of {score * 100}% ")

# KNN method for binary classification


def KNNClassifier(x_train, y_train, x_test, k):

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    y_prediction = neigh.predict(x_test)
    score = metrics.accuracy_score(y_test, y_prediction)
    print(f"The proposed knn model has an accuracy of {score * 100}% ")
    return y_prediction


if __name__ == "__main__":
    # Check whether the data matrix is saved as an existed file
    if os.path.exists(original_mri_matrix_path):
        mri_image_matrix = np.load(original_mri_matrix_path)
    else:
        mri_image_matrix = Data_Pre_Processing.generate_label_of_mri_image_matrix(data_direction, label_path)
    # The process to delete the last column - labels
    x = np.delete(mri_image_matrix, 262144, 1)
    # The process to obtain the labels
    y = mri_image_matrix[:, -1]

    # The process to operate svm_with_pca method
    n_components = 100
    svm_with_pca(x, y, n_components)

    # The process to operate knn method
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    KNNClassifier(x_train, y_train, x_test, 1)
    # The process to explore the optimum value of k
    score_list = []
    for i in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        score_list.append(metrics.accuracy_score(y_test, pred_i))
    # Plot the figure of accuracy and k value
    plt.plot(range(1, 100), score_list, color='pink', linestyle='dashed', marker='o', markerfacecolor='grey',
             markersize=10)
    plt.title("The relationship between accuracy and k value")
    plt.xlabel("k")
    plt.ylabel("The accuracy value")
    plt.show()







