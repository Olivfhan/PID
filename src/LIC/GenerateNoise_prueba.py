import cv2
import numpy as np
import matplotlib.pyplot as plt


def label_regions(im, num_regions): #regiones

    # Builds an 8-connected graph of pixels, with edge weights being the
    # l2 distance between pixel colors in whatever color space they are in.
    # Then, finds the minimum-cost spanning forest with num_regions components.

    H, W, C = im.shape
    J, I = np.meshgrid(np.arange(W), np.arange(H))

    edges = []
    indices = I * W + J

    h_cost = np.sum((im[:, :-1, :] - im[:, 1:, :]) ** 2, axis=2) ** 0.5
    h_edges = list(zip(h_cost.ravel(), indices[:, :-1].ravel(), indices[:, 1:].ravel()))
    edges += h_edges

    v_cost = np.sum((im[:-1, :, :] - im[1:, :, :]) ** 2, axis=2) ** 0.5
    v_edges = list(zip(v_cost.ravel(), indices[:-1, :].ravel(), indices[1:, :].ravel()))
    edges += v_edges

    dr_cost = np.sum((im[:-1, :-1, :] - im[1:, 1:, :]) ** 2, axis=2) ** 0.5
    dr_edges = list(zip(dr_cost.ravel(), indices[:-1, :-1].ravel(), indices[1:, 1:].ravel()))
    edges += dr_edges

    ur_cost = np.sum((im[1:, :-1, :] - im[:-1, 1:, :]) ** 2, axis=2) ** 0.5
    ur_edges = list(zip(ur_cost.ravel(), indices[1:, :-1].ravel(), indices[:-1, 1:].ravel()))
    edges += ur_edges

    edges.sort()

    p, rank, components = np.arange(H * W), np.zeros(H * W), H * W

    def parent(x):
        if p[x] == x:
            return x
        p[x] = parent(p[x])
        return p[x]

    for cost, x, y in edges:
        if components <= num_regions:
            break

        x = parent(x)
        y = parent(y)
        if x == y:
            continue
        if rank[x] > rank[y]:
            x, y = y, x
        if rank[x] == rank[y]:
            rank[y] += 1
        p[x] = y
        components -= 1

    component_labels = dict()
    labels = np.zeros((H, W), dtype=np.int32)
    label_counts = []
    for i in range(H):
        for j in range(W):
            pi = parent(i * W + j)
            if pi not in component_labels:
                component_labels[pi] = len(component_labels)
                label_counts.append(0)
            labels[i, j] = component_labels[pi]
            label_counts[labels[i, j]] += 1

    return labels, label_counts


def generate_noise_image(im_gray, labels, label_counts, #generación de ruido
                         lambda_1=0.7, min_1=0, max_1=255,
                         lambda_2=0.3, min_2=0, max_2=255):
    
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
    
    im_gray = im_gray / 255.0
    H, W = im_gray.shape
    im_noise = np.zeros(im_gray.shape)

    K = len(label_counts)
    R = np.zeros(K)
    for i in range(H):
        for j in range(W):
            R[labels[i, j]] += im_gray[i, j]
    R /= np.array(label_counts)

    T_1 = lambda_1 * (1.0 - im_gray) ** 2.0
    T_2 = lambda_2 * (1.0 - im_gray) ** 2.0
    P = np.random.uniform(0, 1, (H, W))
    C_1 = min_1 * (P <= T_1) + max_1 * (P > T_1)
    C_2 = min_2 * (P <= T_2) + max_2 * (P > T_2)

    I = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            I[i, j] = im_gray[i, j] <= R[labels[i, j]]

    im_noise = C_1 * I + C_2 * (1.0 - I)
    return im_noise


# Cargas de la imagen según el artículo
filepath = "Images/Abdullah_Ahmad_Badawi_0001.jpg"  
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Generar etiquetas 
labels, label_counts = label_regions(img_lab, img_lab.shape[0] * img_lab.shape[1] // 8)
#Generar ruido 
im_noise = generate_noise_image(img_gray, labels, label_counts)

# Mostrar resultado
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Imagen Original")
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Imagen con Ruido")
plt.imshow(im_noise, cmap="gray")
plt.axis("off")

plt.show()
