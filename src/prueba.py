import cv2
import numpy as np
import matplotlib.pyplot as plt

length = 30
dt = 0.2

def generate_white_noise(input_img):
    """
    Genera un ruido blanco a partir de la imagen de entrada.
    La probabilidad de que un píxel sea blanco está proporcional a la intensidad.
    """
    norm = input_img / 255.0  # Normaliza la imagen a [0, 1]
    noise = np.random.rand(*input_img.shape)  # Ruido aleatorio
    white_noise = (noise < norm).astype(np.float32)  # Píxeles blancos donde el ruido es menor que la intensidad
    return white_noise

def compute_vector_field(input_img):
    """
    Calcula el campo vectorial a partir de la imagen de entrada.
    Se utiliza el operador Sobel para calcular los gradientes, se determina el ángulo y se rota 90° para obtener la dirección del trazo.
    """
    # Calcula gradientes en x e y
    grad_x = cv2.Sobel(input_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_img, cv2.CV_32F, 0, 1, ksize=3)
    # Ángulo del gradiente
    angle = np.arctan2(grad_y, grad_x)
    # Rotar el ángulo 90° para obtener la dirección del trazo
    stroke_angle = angle + np.pi / 2
    # Campo vectorial: cada vector es (cos(ángulo), sin(ángulo))
    vx = np.cos(stroke_angle)
    vy = np.sin(stroke_angle)
    vector_field = np.dstack((vx, vy))
    return vector_field

def lic(noise, vector_field, length, dt):
    import numpy as np
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

                # Aplicar contorno periódico usando módulo
                ix0 = int(x) % width
                iy0 = int(y) % height
                # Para el vecino derecho/inferior, se utiliza el módulo para envolver
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

# Ejecución principal
if __name__ == '__main__':
    # Cargar imagen de entrada (asegúrate de que la ruta sea correcta)
    input_img = cv2.imread('Images/Abdullah_Ahmad_Badawi_0001.jpg', cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Paso 1: Genera el ruido blanco
    noise = generate_white_noise(input_img)
    
    # Paso 2: Calcula el campo vectorial basado en los gradientes de la imagen
    vector_field = compute_vector_field(input_img)
    
    # Paso 3: Aplica el LIC para generar el sketch
    lic_result = lic(noise, vector_field, length, dt)
    
    # Mostrar la imagen original, el ruido blanco y el resultado del LIC
    plt.figure(figsize=(15,5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Imagen de Entrada')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noise, cmap='gray')
    plt.title('Ruido Blanco')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(lic_result, cmap='gray')
    plt.title('Resultado LIC')
    plt.axis('off')
    
    plt.show()