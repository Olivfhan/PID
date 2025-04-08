

# Integración del nuevo algoritmo LIC (más fiel) dentro del pipeline anterior

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Parámetros
K = 5
resize_dim = (256, 256)
length = 40  # LIC length
dt = 0.3     # LIC paso
gaussian_blur_ksize = 3

# Paso (a): Cargar imagen
input_image = cv2.imread("Images/raw/Chakib_Khelil_0001.jpg", cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, resize_dim)
input_blurred = cv2.GaussianBlur(input_image, (gaussian_blur_ksize, gaussian_blur_ksize), 0)

# Paso (b): Generar ruido blanco ponderado por intensidad
def generate_white_noise(input_img):
    norm = input_img / 255.0
    noise = np.random.rand(*input_img.shape)
    white_noise = (noise < norm).astype(np.float32)
    return white_noise

white_noise = generate_white_noise(input_image)

# Paso (c): Segmentación
Z = input_image.reshape((-1, 1)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented_image = labels.flatten().reshape(input_image.shape)

# Paso (d): Bordes
segmented_uint8 = (segmented_image * (255 / segmented_image.max())).astype(np.uint8)
edges = cv2.Canny(segmented_uint8, 30, 100)
inverted_edges = cv2.bitwise_not(edges)

# Paso (e): Campo vectorial ortogonal al gradiente
def compute_vector_field(input_img):
    grad_x = cv2.Sobel(input_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_img, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(grad_y, grad_x)
    stroke_angle = angle + np.pi / 2
    vx = np.cos(stroke_angle)
    vy = np.sin(stroke_angle)
    vector_field = np.dstack((vx, vy))
    return vector_field

vector_field = compute_vector_field(input_blurred)

# Paso (f): LIC con integración direccional
def lic(noise, vector_field, length, dt):
    height, width = noise.shape
    lic_image = np.zeros_like(noise)

    for i in range(height):
        for j in range(width):
            x, y = j, i
            total = 0
            weight = 0
            for t in range(length):
                vx, vy = vector_field[i, j]
                x += vx * dt
                y += vy * dt

                ix0 = int(x) % width
                iy0 = int(y) % height
                ix1 = (ix0 + 1) % width
                iy1 = (iy0 + 1) % height
                dx = x - int(x)
                dy = y - int(y)

                value = (noise[iy0, ix0] * (1 - dx) * (1 - dy) +
                         noise[iy0, ix1] * dx       * (1 - dy) +
                         noise[iy1, ix0] * (1 - dx) * dy       +
                         noise[iy1, ix1] * dx       * dy)
                total += value
                weight += 1

            lic_image[i, j] = total / weight if weight > 0 else 0
    return lic_image

lic_image = lic(white_noise, vector_field, length=length, dt=dt)
lic_image_uint8 = (lic_image * 255).astype(np.uint8)

# Paso (g): Añadir bordes invertidos
lic_float = lic_image_uint8.astype(np.float32)
edges_float = inverted_edges.astype(np.float32)

outlined_image = cv2.addWeighted(lic_float, 0.85, edges_float, 0.15, 0)
outlined_image = np.clip(outlined_image, 0, 255).astype(np.uint8)


# Paso (h): Cargar textura y proyectarla (omitida por ahora)
paper_texture = cv2.imread("Images/textures/textura_lapiz.jpg", cv2.IMREAD_GRAYSCALE)
paper_texture = cv2.resize(paper_texture, input_image.shape[::-1])

# Paso (i): Composición final
final_drawing = cv2.addWeighted(outlined_image.astype(np.uint8), 0.7, paper_texture, 0.3, 0)

# Mostrar todo el pipeline actualizado
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

titles = [
    "a) Imagen de entrada",
    "b) Ruido blanco",
    "c) Segmentación (KMeans)",
    "d) Bordes",
    "e) Campo de vectores",
    "f) LIC",
    "g) Con bordes invertidos",
    "h) Textura de lapiz",
    "i) Dibujo final"
]

images = [
    input_image,
    white_noise,
    segmented_image * (255 // K),
    edges,
    ((np.arctan2(vector_field[...,1], vector_field[...,0]) + np.pi) / (2*np.pi) * 255).astype(np.uint8),
    lic_image_uint8,
    outlined_image.astype(np.uint8),
    paper_texture,
    final_drawing
]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
