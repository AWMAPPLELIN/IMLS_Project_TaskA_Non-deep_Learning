import os
import cv2.cv2 as cv2
import pandas as pd
import numpy as np


# The process to convert the 3000 mri images of jpg format into a matrix with the input to be the data direction
# The process to return a matrix with 1 row x 262145 column
# The first 262144 columns are the characteristics of mri images and the last column is the label
def generate_label_of_mri_image_matrix(data_direction, label_path):
    # The process to quickly create an array
    mri_image_matrix = np.empty((1, 262145), int)
    df = pd.read_csv(label_path)
    df = df.set_index('file_name')
    # The process to read the image's matrix
    for fileName in os.listdir(data_direction):
        print("This is img " + fileName)
        full_path = os.path.join(data_direction, fileName)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    # The process to check the class of different images
        if df.loc[fileName, "label"] == "no_tumor":
            image_class = 0
        else:
            image_class = 1

    # The process to convert original image matrix with 512 row x 512 column to a matrix with 1 row x 262144 column
    # Then it is the process to add the class to the matrix and it turns out to be a matrix with 1 row x 262145 column
        image_row_vector = image.flatten()
        data_vec = np.append(image_row_vector, image_class)
        mri_image_matrix = np.append(mri_image_matrix, [data_vec], axis=0)
    return mri_image_matrix


if __name__ == "__main__":
    data_direction = "dataset-0/image"
    label_path = "dataset-0/label.csv"
    matrix_file_name = "MRI_Image_Matrix.npy"

    if os.path.exists(matrix_file_name):
        mri_image_matrix = np.load(matrix_file_name)
    else:
        mri_image_matrix = generate_label_of_mri_image_matrix(data_direction, label_path)
        np.save(matrix_file_name, mri_image_matrix)
    print(mri_image_matrix.shape)
