import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Inicializar el cuadro de diálogo para seleccionar una imagen
Tk().withdraw()  # No queremos una GUI completa, por lo que evitamos que aparezca la ventana raíz
image_file = askopenfilename(title="Selecciona una imagen", filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")])
if not image_file:
    print("No se seleccionó ningún archivo. Saliendo del programa.")
    exit()

# Configuración de los archivos del modelo y la ruta de la imagen
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
threshold = 0.1  # Umbral para la detección de puntos clave

# Carga de la red neuronal
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

# Lectura de la imagen
frame = cv2.imread(image_file)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Preparación del blob a partir de la imagen
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()

# Mapeo de los puntos clave de interés (Modelo COCO)
keypoints_map = {
    'nose': 0,
    'left_eye': 14,  # Confirmar que estos índices son correctos para el modelo COCO utilizado
    'right_eye': 15,
    'left_shoulder': 5,
    'right_shoulder': 2,
    'left_hip': 11,
    'right_hip': 8,
    'left_knee': 13,
    'right_knee': 10,
    'left_ankle': 15,
    'right_ankle': 16
}

# Detección de puntos clave
points = {}
for key, idx in keypoints_map.items():
    probMap = output[0, idx, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (frameWidth * point[0]) / output.shape[3]
    y = (frameHeight * point[1]) / output.shape[2]
    if prob > threshold:
        points[key] = (int(x), int(y))
    else:
        points[key] = None

# Función para analizar la postura (implementación simplificada)
def analyze_posture(points):
    # Analizar la alineación de los hombros y las caderas para determinar si la postura es buena
    # Una postura se considera buena si los hombros y las caderas están alineados horizontalmente
    if points['left_shoulder'] and points['right_shoulder'] and points['left_hip'] and points['right_hip']:
        # Calcular la diferencia en la altura de los hombros y caderas
        shoulder_difference = abs(points['left_shoulder'][1] - points['right_shoulder'][1])
        hip_difference = abs(points['left_hip'][1] - points['right_hip'][1])
        
        # Si la diferencia en la altura de los hombros y caderas es pequeña, la postura es buena
        if shoulder_difference < 20 and hip_difference < 20:
            return 'Buena postura'
        else:
            return 'Mala postura'
    else:
        return 'Datos insuficientes para analizar la postura'

# Analizar la postura
posture = analyze_posture(points)
print("Postura detectada:", posture)

# Definición de las conexiones correctas entre los puntos clave
POSE_PAIRS = [
    ["left_shoulder", "left_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_shoulder", "right_hip"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
    ["nose", "left_eye"],
    ["nose", "right_eye"],
    ["nose", "left_shoulder"],
    ["nose", "right_shoulder"]
]

# Dibujar los esqueletos
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    if partA in points and partB in points and points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

# Dibujar círculos en los puntos clave
for point in points.values():
    if point:
        cv2.circle(frame, point, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# Redimensionar la imagen para mostrar si es demasiado grande para la pantalla
scale_percent = 100  # Porcentaje de escala; 100 significa sin escala. Ajusta esto si es necesario.
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar imagen
resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Mostrar la imagen redimensionada en una ventana que puede redimensionarse
cv2.namedWindow('Detección de Puntos Clave', cv2.WINDOW_NORMAL)
cv2.imshow('Detección de Puntos Clave', resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
