# -*- coding: utf-8 -*-
"""
Algoritmo mejorado de LIC para boceto a lápiz realista con XDoG manual corregido
"""
import numpy as np
import cv2
from skimage.segmentation import slic
import matplotlib.pyplot as plt

# Parámetros generales
resize_dim = (256, 256)
K = 200  # número de regiones para segmentación más fino
gradient_thresh = 0.2  # umbral para zonas sin textura
default_angle = np.deg2rad(45)
length_fine = 10  # longitud corta para detalles
length_coarse = 40  # longitud larga para sombreado general
dt = 0.3
# Parámetros de filtro bilateral
bilateral_d = 9
bilateral_sigma_color = 75
bilateral_sigma_space = 75
# Parámetros XDoG manual
xdog_sigma = 0.5
xdog_k = 1.6
xdog_eps = 0.02
xdog_phi = 200.0

# Funciones auxiliares
def preprocess(img):
    """Aplica filtro bilateral para suavizar sin perder bordes"""
    return cv2.bilateralFilter(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

# Implementación manual de XDoG
def xdog(img, sigma=xdog_sigma, k=xdog_k, eps=xdog_eps, phi=xdog_phi):
    f = img.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(f, (0, 0), sigma)
    g2 = cv2.GaussianBlur(f, (0, 0), sigma * k)
    dog = g1 - g2
    x = np.where(dog >= eps,
                 1.0,
                 1.0 + np.tanh(phi * (dog - eps)))
    edges = ((1.0 - x) * 255).astype(np.uint8)
    return edges

# Carga y preprocesado
def load_and_preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, resize_dim)
    img_pre = preprocess(img)
    return img, img_pre

# Cálculo del campo vectorial y magnitud

def compute_gradient_field(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx) + np.pi / 2
    vx = np.cos(theta)
    vy = np.sin(theta)
    field = np.dstack((vx, vy))
    return field, mag

# Suavizado tipo ETF

def edge_tangent_flow(field, iterations=5, kernel_size=5):
    f = field.copy()
    for _ in range(iterations):
        fx = cv2.GaussianBlur(f[..., 0], (kernel_size, kernel_size), 0)
        fy = cv2.GaussianBlur(f[..., 1], (kernel_size, kernel_size), 0)
        f = np.dstack((fx, fy))
        norm = np.linalg.norm(f, axis=2, keepdims=True) + 1e-8
        f = f / norm
    return f

# Ajuste por región de superpíxeles

def orient_by_region(field, segments):
    f = field.copy()
    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        mean_vec = f[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec /= norm
            f[mask] = (f[mask] + mean_vec) * 0.5
    norm = np.linalg.norm(f, axis=2, keepdims=True) + 1e-8
    return f / norm

# Orientación por defecto en zonas sin textura

def assign_default_orientation(field, mag):
    f = field.copy()
    default_vec = np.array([np.cos(default_angle), np.sin(default_angle)])
    mask = (mag < gradient_thresh)
    f[mask] = default_vec
    return f

# Generación de ruido multicapa

def generate_noise(img, gaussian=False):
    norm = img.astype(np.float32) / 255.0
    if gaussian:
        return np.random.randn(*img.shape) * norm
    rnd = np.random.rand(*img.shape)
    return (rnd < norm).astype(np.float32)

# Convolución LIC con kernel

def lic_convolve(noise, field, length, dt, use_gaussian_kernel=True):
    h, w = noise.shape
    result = np.zeros_like(noise, dtype=np.float32)
    t = np.arange(-length, length + 1)
    if use_gaussian_kernel:
        sigma = length / 3.0
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
    else:
        kernel = np.ones_like(t, dtype=np.float32)
    kernel /= kernel.sum()
    for i in range(h):
        for j in range(w):
            acc = 0.0
            wsum = 0.0
            for idx, wt in enumerate(kernel):
                step = idx - length
                x = j + field[i, j, 0] * dt * step
                y = i + field[i, j, 1] * dt * step
                xi = int(np.clip(x, 0, w - 1))
                yi = int(np.clip(y, 0, h - 1))
                acc += noise[yi, xi] * wt
                wsum += wt
            result[i, j] = acc / (wsum + 1e-8)
    return result

# Pipeline principal

def main(image_path, paper_texture_path):
    img, img_pre = load_and_preprocess(image_path)
    # XDoG para contornos
    edges = xdog(img_pre)
    inv_edges = cv2.bitwise_not(edges)
    # Segmentación
    segments = slic(img_pre, n_segments=K, compactness=10, channel_axis=None)
    # Campo vectorial
    field0, mag = compute_gradient_field(img_pre)
    field1 = edge_tangent_flow(field0)
    field2 = orient_by_region(field1, segments)
    field_final = assign_default_orientation(field2, mag)
    # Ruido y LIC
    noise_coarse = generate_noise(img_pre, gaussian=False)
    noise_fine = generate_noise(img_pre, gaussian=True)
    lic_coarse = lic_convolve(noise_coarse, field_final, length_coarse, dt)
    lic_fine = lic_convolve(noise_fine, field_final, length_fine, dt)
    lic_comb = cv2.addWeighted(lic_coarse, 0.6, lic_fine, 0.4, 0)
    lic_uint8 = (lic_comb * 255).astype(np.uint8)
    # Superponer bordes en float32
    lic_float = lic_uint8.astype(np.float32)
    # Cargar y convertir textura a float32 para coincidir
    paper = cv2.imread(paper_texture_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    paper = cv2.resize(paper, img.shape[::-1])
    outlined = cv2.addWeighted(lic_float, 0.85, inv_edges.astype(np.float32), 0.15, 0)
    final = cv2.addWeighted(outlined, 0.7, paper, 0.3, 0).astype(np.uint8)
    # Mostrar resultados
    titles = ["Entrada", "Preprocesada", "XDoG", "Segmentos",
              "LIC Coarse", "LIC Fine", "Combinado", "Bordes Superpuestos", "Resultado Final"]
    imgs = [img, img_pre, edges, segments * (255 // K),
            (lic_coarse * 255).astype(np.uint8), (lic_fine * 255).astype(np.uint8),
            lic_uint8, outlined.astype(np.uint8), final]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, im, t in zip(axes.flatten(), imgs, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main('Images/raw/Chakib_Khelil_0001.jpg', 'Images/textures/textura_lapiz.jpg')
