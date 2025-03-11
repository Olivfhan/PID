import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo, se debe tener cuidado con los valores ya que un valor muy alto puede hacer que el algoritmo tarde mucho en ejecutarse
length = 40
dt = 0.3

def generate_white_noise(input_img):
    norm = input_img / 255.0  # Normaliza la imagen a [0, 1]
    noise = np.random.rand(*input_img.shape)  # Ruido aleatorio
    white_noise = (noise < norm).astype(np.float32)  # Píxeles blancos donde el ruido es menor que la intensidad
    return white_noise

def compute_vector_field(input_img):
    # Calcula gradientes en x e y
    grad_x = cv2.Sobel(input_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_img, cv2.CV_32F, 0, 1, ksize=3)
    # Ángulo del gradiente
    angle = np.arctan2(grad_y, grad_x)
    stroke_angle = angle + np.pi / 2  # Rotar 90° para la dirección del trazo
    vx = np.cos(stroke_angle)
    vy = np.sin(stroke_angle)
    vector_field = np.dstack((vx, vy))
    return vector_field

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

                # Aplicar contorno periódico
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

if __name__ == '__main__':
    input_img = cv2.imread('Images/raw/Chakib_Khelil_0001.jpg', cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Paso 1: Genera el ruido blanco
    noise = generate_white_noise(input_img)
    
    # Paso 2: Calcula el campo vectorial
    vector_field = compute_vector_field(input_img)
    
    # Paso 3: Aplica el LIC
    lic_result = lic(noise, vector_field, length, dt)
    
    # Paso 4: Detectar contornos
    edges = cv2.Canny(input_img, 100, 200)
    
    # Paso 5: Añadir contornos a la imagen final
    final_result = cv2.addWeighted(lic_result, 0.8, edges.astype(np.float32), 0.2, 0)
    
    # Mostrar resultados
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    '''
    plt.subplot(1, 3, 2)
    plt.imshow(noise, cmap='gray')
    plt.title('Ruido Blanco')
    plt.axis('off')
    '''

    plt.subplot(1, 3, 2)
    plt.imshow(lic_result, cmap='gray')
    plt.title('Resultado LIC')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_result, cmap='gray')
    plt.title('Resultado Final (con contornos)')
    plt.axis('off')
    
    plt.show()
