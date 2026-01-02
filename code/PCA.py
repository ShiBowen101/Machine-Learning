import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

n = 100


def color_to_grayscale(image_path):
    """
    Convert a color image to grayscale and return the grayscale image as a numpy array.

    Parameters:
    - image_path: str, the path to the color image file

    Returns:
    - grayscale_array: numpy array, the grayscale image as a 2D numpy array
    """
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_img = img.convert('L')

    # Convert the grayscale image to a numpy array
    grayscale_array = np.array(grayscale_img)

    return grayscale_array


# Example usage:
# grayscale_matrix = color_to_grayscale('path_to_color_image.jpg')


def perform_pca(matrix, n_components):
    """
    Perform PCA analysis on a matrix.

    Parameters:
    - matrix: numpy array, the input matrix to be analyzed
    - n_components: int, the number of principal components to retain

    Returns:
    - pca_results: numpy array, the transformed matrix after PCA analysis
    """
    # Standardize the matrix
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(matrix)

    # Perform PCA analysis
    pca = PCA(n_components=n_components)
    pca.fit(standardized_matrix)
    pca_results = pca.transform(standardized_matrix)

    return pca_results


# Example usage:
# matrix = np.random.rand(100, 100)  # Example matrix
# pca_transformed_matrix = perform_pca(matrix, n_components=50)  # Retain 50 principal components


gary_image = color_to_grayscale('cat.jpg')
return_vector = perform_pca(gary_image, n)


def pca_reduction(pca_results, original_matrix, n_components):
    """
    Reverse the PCA transformation on the results and return the reconstructed matrix.

    Parameters:
    - pca_results: numpy array, the PCA transformed matrix
    - original_matrix: numpy array, the original matrix before PCA
    - n_components: int, the number of principal components used in the PCA transformation

    Returns:
    - reconstructed_matrix: numpy array, the reconstructed matrix after PCA inverse transformation
    """
    # Invert the PCA transformation
    pca = PCA(n_components=n_components)
    pca.fit(original_matrix)
    reconstructed_matrix = pca.inverse_transform(pca_results)

    return reconstructed_matrix


# Example usage:
# pca_transformed_matrix = perform_pca(matrix, n_components=50)  # Retain 50 principal components
# original_matrix = ...  # Your original matrix
# reconstructed_matrix = pca_reduction(pca_transformed_matrix, original_matrix, n_components=50)
return_image = pca_reduction(return_vector, gary_image, n)


def matrix_to_image_and_display(matrix):
    """
    Convert a grayscale matrix to an image and display it.

    Parameters:
    - matrix: numpy array, the grayscale matrix to be converted

    Returns:
    - None
    """
    # Normalize the matrix to be in the range [0, 255]
    matrix = np.clip(matrix * 255, 0, 255).astype(np.uint8)

    # Convert the matrix to an image
    image = Image.fromarray(matrix)

    # Display the image
    image.show()


# Example usage:
# matrix = ...  # Your grayscale matrix
# matrix_to_image_and_display(matrix)

photo = matrix_to_image_and_display(return_image)
print(return_image)
print(gary_image)