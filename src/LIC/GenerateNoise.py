import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_noise_image(im_gray, labels, label_counts, lambda_1=0.7, min_1=0, max_1=255, lambda_2=0.3, min_2=0, max_2=255):
    """
    Generates a speckle noise image based on region labels and grayscale values.
    
    Args:
        im_gray (np.ndarray): 2D array of grayscale image values.
        labels (np.ndarray): 2D array of pixel region labels.
        label_counts (np.ndarray): Array containing the frequency of region labels.
        lambda_1 (float): Threshold 1 tuning parameter.
        min_1 (float): Threshold 1 low output value.
        max_1 (float): Threshold 1 high output value.
        lambda_2 (float): Threshold 2 tuning parameter.
        min_2 (float): Threshold 2 low output value.
        max_2 (float): Threshold 2 high output value.
    
    Returns:
        np.ndarray: 2D array representing the generated noise image.
    """
    im_gray = im_gray.astype(np.float64) / 255.0
    H, W = im_gray.shape
    im_noise = np.zeros((H, W), dtype=np.float64)
    
    # Compute region means
    N = len(label_counts)
    R = np.zeros(N, dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            R[labels[i, j]] += im_gray[i, j]
    
    R /= label_counts
    
    # Compute colors if thresholds 1 and 2 are used
    T_1 = lambda_1 * (1.0 - im_gray) ** 2.0
    T_2 = lambda_2 * (1.0 - im_gray) ** 2.0
    P = np.random.rand(H, W)
    C_1 = np.where(P <= T_1, min_1, max_1)
    C_2 = np.where(P <= T_2, min_2, max_2)
    
    # Select pixels to be colored with threshold 1
    I = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            I[i, j] = im_gray[i, j] <= R[labels[i, j]]
    
    # Color pixels by the correct threshold
    im_noise = np.where(I, C_1, C_2)
    
    return im_noise




# Cargar la imagen en escala de grises
image_path = "Images/Abdullah_Ahmad_Badawi_0001.jpg"  # Reemplaza con la ruta de tu imagen
im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalizar a rango [0,1]

# Generar etiquetas y conteo de regiones (simulación básica)
H, W = im_gray.shape
labels = np.random.randint(0, 5, (H, W))  # Simulación de segmentación con 5 regiones
label_counts = np.bincount(labels.ravel(), minlength=5)

# Generar imagen con ruido
noise_image = generate_noise_image(im_gray, labels, label_counts)

# Mostrar resultado
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Imagen Original")
plt.imshow(im_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Imagen con Ruido")
plt.imshow(noise_image, cmap="gray")
plt.axis("off")

plt.show()