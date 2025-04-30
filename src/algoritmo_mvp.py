import numpy as np
import cv2
import matplotlib.pyplot as plt
# Asegúrate de que skimage está instalado: pip install scikit-image
# from skimage.segmentation import felzenszwalb # Ya no se usa activamente
# Asegúrate de que etf_utils.py está en el mismo directorio o en el PYTHONPATH
from etf_utils import compute_etf_field 

# --- Parámetros Configurables ---
KERNEL_SIZE_ETF = 5        # Tamaño del kernel para ETF
RESIZE_DIM = (512, 512)  # Redimensionar imagen para procesar
LIC_LENGTH = 30          # Longitud de la línea para LIC (afecta longitud del trazo)
LIC_DT = 0.3             # Paso de integración para LIC
LIC_SIGMA = 3.0          # Sigma para el kernel Gaussiano en LIC
BILATERAL_D = 7          # Diámetro del vecindario para filtro bilateral (reducido?)
BILATERAL_SIGMA_COLOR = 50 # Reducido para ignorar más textura fina?
BILATERAL_SIGMA_SPACE = 50 # Reducido para suavizar menos espacialmente?
ETF_ITERATIONS = 3       # Iteraciones para refinar ETF
GRADIENT_THRESHOLD = 0.1 # Umbral de magnitud de gradiente para detectar zonas lisas
DEFAULT_ORIENTATION_ANGLE = 45 # Ángulo por defecto en zonas lisas (grados)
# --- Parámetros de Canny ---
CANNY_THRESHOLD1 = 15   # Umbral bajo para Canny (probar ajustando estos)
CANNY_THRESHOLD2 = 45   # Umbral alto para Canny 
# --- Parámetros de Composición ---
EDGE_GRAY_VALUE = 45     # Valor de gris para los bordes (0=negro, 255=blanco)
PAPER_TEXTURE_WEIGHT = 0.2 # Peso de la textura de papel (0 para no usarla)
# --- Rutas ---
INPUT_IMAGE_PATH = "Images/raw/Chakib_Khelil_0001.jpg" # CAMBIAR por la ruta de tu imagen
PAPER_TEXTURE_PATH = "Images/textures/textura_lapiz.jpg" # CAMBIAR si usas otra textura
OUTPUT_IMAGE_PATH = "Output/sketch_mejorado_v4.png" # Nuevo nombre para la salida

# --- Utilidades (generate_white_noise_proportional, enforce_default_orientation, lic_gaussian - sin cambios) ---
def generate_white_noise_proportional(input_img_gray):
    """Genera ruido blanco donde la probabilidad de blanco es proporcional a la intensidad."""
    if input_img_gray.max() > 1:
        norm = input_img_gray.astype(np.float32) / 255.0
    else:
        norm = input_img_gray.astype(np.float32)
    noise = np.random.rand(*input_img_gray.shape)
    white_noise = (noise > norm).astype(np.float32) 
    return white_noise

def enforce_default_orientation(vector_field, gradient_magnitude, threshold, angle_deg=45):
    """Asigna una orientación por defecto a los vectores en zonas de bajo gradiente."""
    angle_rad = np.deg2rad(angle_deg)
    default_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)]) 
    enforced_field = vector_field.copy()
    mask = gradient_magnitude < threshold
    norm = np.linalg.norm(default_vector)
    if norm > 0:
        default_vector /= norm
    enforced_field[mask] = default_vector
    return enforced_field

def lic_gaussian(noise, vector_field, length=20, dt=0.3, sigma=5.0):
    """Aplica Line Integral Convolution usando un kernel Gaussiano."""
    h, w = noise.shape[:2] 
    output = np.zeros_like(noise, dtype=np.float32)
    
    if vector_field.shape[2] != 2:
        raise ValueError("El campo vectorial debe tener 2 canales (x, y)")

    kernel_radius = length // 2
    kernel_size = length 

    t = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    gaussian_kernel = np.exp(-(t**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum() 

    for y0 in range(h):
        for x0 in range(w):
            xc, yc = float(x0), float(y0)
            streamline_values = []
            
            # Adelante
            x, y = xc, yc
            for _ in range(kernel_radius):
                iy, ix = int(round(y)), int(round(x))
                if not (0 <= iy < h and 0 <= ix < w): break
                vy, vx = vector_field[iy, ix] 
                x_new, y_new = x + vx * dt, y + vy * dt
                x, y = x_new, y_new
                iy_new, ix_new = int(round(y)), int(round(x))
                if not (0 <= iy_new < h and 0 <= ix_new < w): break
                streamline_values.append(noise[iy_new, ix_new])

            # Atrás
            x, y = xc, yc
            backward_values = [] 
            for _ in range(kernel_radius):
                iy, ix = int(round(y)), int(round(x))
                if not (0 <= iy < h and 0 <= ix < w): break
                vy, vx = vector_field[iy, ix]
                x_new, y_new = x - vx * dt, y - vy * dt
                x, y = x_new, y_new
                iy_new, ix_new = int(round(y)), int(round(x))
                if not (0 <= iy_new < h and 0 <= ix_new < w): break
                backward_values.append(noise[iy_new, ix_new])

            combined_values = backward_values[::-1] + [noise[y0, x0]] + streamline_values
            
            actual_len = len(combined_values)
            if actual_len > 0:
                 kernel_start = max(0, kernel_size // 2 - actual_len // 2)
                 kernel_end = min(kernel_size, kernel_start + actual_len)
                 current_kernel = gaussian_kernel[kernel_start:kernel_end]
                 if len(current_kernel) == actual_len:
                     current_kernel /= current_kernel.sum() 
                     value_sum = np.sum(np.array(combined_values) * current_kernel)
                     output[y0, x0] = value_sum
                 else:
                      output[y0, x0] = np.mean(combined_values)
            else:
                output[y0, x0] = noise[y0, x0] 

    return output

# --- NUEVA Función de Detección de Bordes ---
def canny_edges(image, threshold1, threshold2):
    """Detecta bordes usando Canny."""
    # Canny requiere uint8
    if image.max() > 1 and image.dtype != np.uint8:
         image_uint8 = image.astype(np.uint8) 
    elif image.max() <= 1:
         image_uint8 = (image * 255).astype(np.uint8)
    else:
         image_uint8 = image
         
    edges = cv2.Canny(image_uint8, threshold1, threshold2)
    # edges: 255=borde, 0=fondo.
    return edges

# --- Pipeline Principal ---
print("Cargando imagen...")
input_image_color = cv2.imread(INPUT_IMAGE_PATH)
if input_image_color is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {INPUT_IMAGE_PATH}")
input_image_color = cv2.resize(input_image_color, RESIZE_DIM)
input_image_gray = cv2.cvtColor(input_image_color, cv2.COLOR_BGR2GRAY)

print("Aplicando preprocesamiento (Filtro Bilateral)...")
# Preprocesamiento: Filtro Bilateral (ajusta parámetros si es necesario)
input_blurred = cv2.bilateralFilter(input_image_gray, BILATERAL_D, 
                                    BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

print("Detectando bordes (Canny)...")
# Detección de bordes con Canny (Bordes=255, Fondo=0)
edges_canny = canny_edges(input_blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

print("Calculando campo vectorial ETF...")
vector_field_raw = compute_etf_field(input_blurred, iterations=ETF_ITERATIONS, kernel_size=KERNEL_SIZE_ETF) 

print("Calculando magnitud del gradiente...")
grad_x = cv2.Sobel(input_blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(input_blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)
gradient_magnitude_norm = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX) 

print("Aplicando orientación por defecto en zonas lisas...")
vector_field_corrected = enforce_default_orientation(vector_field_raw, 
                                                    gradient_magnitude_norm, 
                                                    GRADIENT_THRESHOLD, 
                                                    angle_deg=DEFAULT_ORIENTATION_ANGLE)

print("Generando ruido blanco proporcional...")
white_noise = generate_white_noise_proportional(input_blurred)

print("Aplicando Line Integral Convolution (LIC)...")
lic_image = lic_gaussian(white_noise, vector_field_corrected, length=LIC_LENGTH, dt=LIC_DT, sigma=LIC_SIGMA)
lic_image_norm = cv2.normalize(lic_image, None, 0, 255, cv2.NORM_MINMAX)
lic_image_uint8 = lic_image_norm.astype(np.uint8)
lic_image_inv = cv2.bitwise_not(lic_image_uint8) # Trazos oscuros, fondo blanco

print("Componiendo resultado con bordes grises...")
# Crear imagen de bordes grises
# Empezar con una imagen blanca
gray_edges_map = np.full_like(input_blurred, 255, dtype=np.uint8) 
# Poner los bordes detectados por Canny en el color gris especificado
gray_edges_map[edges_canny == 255] = EDGE_GRAY_VALUE 

# Combinar LIC invertido con los bordes grises usando cv2.min
# Esto asegura que el resultado no sea más oscuro que el borde gris
# y no más oscuro que el trazo LIC
composite_gray_edges = cv2.min(lic_image_inv, gray_edges_map)

# Opcional: Añadir textura de papel
if PAPER_TEXTURE_WEIGHT > 0:
    print("Añadiendo textura de papel...")
    paper_texture = cv2.imread(PAPER_TEXTURE_PATH, cv2.IMREAD_GRAYSCALE)
    if paper_texture is None:
        print(f"Advertencia: No se pudo cargar la textura de papel desde {PAPER_TEXTURE_PATH}")
        final_drawing_float = composite_gray_edges.astype(np.float32) / 255.0
    else:
        paper_texture = cv2.resize(paper_texture, RESIZE_DIM)
        paper_texture_float = paper_texture.astype(np.float32) / 255.0
        composite_float = composite_gray_edges.astype(np.float32) / 255.0
        # Mezclar (aquí puedes ajustar cómo se combina con la textura)
        # addWeighted puede funcionar bien
        final_drawing_float = cv2.addWeighted(composite_float, 1.0 - PAPER_TEXTURE_WEIGHT,
                                            paper_texture_float, PAPER_TEXTURE_WEIGHT, 0)
else:
    final_drawing_float = composite_gray_edges.astype(np.float32) / 255.0

# Convertir resultado final a uint8
final_drawing = np.clip(final_drawing_float * 255, 0, 255).astype(np.uint8)

print(f"Guardando resultado en {OUTPUT_IMAGE_PATH}...")
cv2.imwrite(OUTPUT_IMAGE_PATH, final_drawing)

# --- Mostrar resultados intermedios ---
print("Mostrando resultados...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

titles = [
    "a) Imagen Original", "b) Preprocesada (Bilateral)", "c) Bordes (Canny)", 
    "d) Bordes Grises (Comp.)", "e) Ruido Proporcional", "f) LIC Invertido", 
    "g) LIC + Bordes Grises", "h) Dibujo Final (+Textura)"
]

# Visualización del campo vectorial (igual que antes)
angle = np.arctan2(vector_field_corrected[..., 1], vector_field_corrected[..., 0])
hue = (angle + np.pi) / (2 * np.pi) 
saturation = np.ones_like(hue)
value = cv2.normalize(gradient_magnitude_norm, None, 0.2, 1.0, cv2.NORM_MINMAX) 
hsv = cv2.cvtColor(np.stack([hue*179, saturation*255, value*255], axis=-1).astype(np.uint8), cv2.COLOR_HSV2BGR)


images = [
    cv2.cvtColor(input_image_color, cv2.COLOR_BGR2RGB), 
    input_blurred,
    edges_canny, # Mostrar bordes Canny (blancos sobre negro)
    gray_edges_map, # Mostrar mapa de bordes grises
    white_noise,
    lic_image_inv,
    composite_gray_edges, # Mostrar resultado antes de textura
    final_drawing
]

for ax, img, title in zip(axes, images, titles):
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255) # Asegurar rango para escala de grises
    else:
        ax.imshow(img) 
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Proceso completado.")