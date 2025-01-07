from PIL import Image
import numpy as np
import os
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import median
from skimage.morphology import closing, square

def load_image(file_path):
    """Load an image from a file, converting it to grayscale if needed."""
    try:
        img = Image.open(file_path)
        img = img.convert('L')  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image using anisotropic diffusion filtering and normalization."""
    # Apply anisotropic diffusion filtering
    denoised_image = denoise_tv_chambolle(image, weight=0.1)
    # Apply median filtering to remove salt-and-pepper noise
    denoised_image = median(denoised_image, square(3))
    # Normalize the image
    normalized_image = (denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image))
    return normalized_image

def fuzzy_c_means_segmentation(image, n_clusters=3, m=2, error=0.005, max_iter=100):
    """Apply Fuzzy C-Means clustering to the image."""
    # Flatten the image to 1D (each pixel is a feature)
    img_flat = image.flatten()
    # Reshape for compatibility with the FCM function
    img_flat = img_flat[None, :]  # shape becomes (1, num_pixels)
    # Apply fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(img_flat, n_clusters, m, error=error, maxiter=max_iter)
    # Reshape the fuzzy partition matrix to the original image size
    segmentation = np.argmax(u, axis=0).reshape(image.shape)
    return segmentation

def double_estimation(initial_segmentation, image, n_clusters=3, m=2, error=0.005, max_iter=100):
    """Refine the initial segmentation using a double estimation technique."""
    # This is a placeholder for the double estimation process
    # You can implement additional refinement steps here
    # For example, re-clustering with additional constraints or post-processing
    refined_segmentation = fuzzy_c_means_segmentation(image, n_clusters, m, error, max_iter)
    # Apply morphological closing to refine the segmentation
    refined_segmentation = closing(refined_segmentation, square(3))
    return refined_segmentation

def save_segmented_image(segmentation, output_path):
    """Save the segmented image to the specified path."""
    plt.imsave(output_path, segmentation, cmap='jet')

def main():
    base_directory = r"C:\Users\mavee\Downloads\Dataset\Brain Tumor MRI images"
    output_base_directory = r"C:\Users\mavee\Downloads\Processed_Images"
    subdirectories = ['Tumor', 'Healthy']

    for subdir in subdirectories:
        input_directory = os.path.join(base_directory, subdir)
        output_directory = os.path.join(output_base_directory, subdir)

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")
                    image = load_image(file_path)
                    if image is not None:
                        # Preprocess the image
                        preprocessed_image = preprocess_image(image)
                        # Initial segmentation using FCM
                        initial_segmentation = fuzzy_c_means_segmentation(preprocessed_image)
                        # Refine the segmentation using double estimation
                        final_segmentation = double_estimation(initial_segmentation, preprocessed_image)
                        # Save the segmented image
                        output_file_path = os.path.join(output_directory, file)
                        save_segmented_image(final_segmentation, output_file_path)
                        print(f"Saved segmented image to {output_file_path}")

if __name__ == "__main__":
    main()
