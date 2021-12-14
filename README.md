# IMLS_Project_TaskA
The main work of this task is to use non-deep learning methods including SVM(PCA) and KNN to do binary classification operations of MRI images.
As for the files in this project, the function of the file: Data_pre_processing is for processing the image data by converting each image to a matrix and making labels for different images;
the function of the file: PCA_Feature_Extraction is for PCA operation and reducing and the dimensions for the whole dataset; the function of the main file is the trunk of the whole task and it includes the realization of two binary classification methods: SVM+PCA and KNN.
The running environment for this is python 3 and you could download all the files and create a virtual environment to run all of them in the software PyCharm. In the main file, it calls all other files and you could obtain the final result from it while some pre-processed datasets including MRI_Image_Matrix and PCA_n_components could also be obtained from the training process for the purpose of fast training at the next time.
Necessary installation of packages and header files include os, pandas, NumPy, pickle, Scikit-learn, matplotlib, OpenCV, etc. 
