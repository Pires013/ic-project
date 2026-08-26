import cv2
import time
import math

from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

def calcular_ear(olho_pontos):
    """
    Calcula o Eye Aspect Ratio (EAR) com base em 6 pontos do olho.
    """
    p1, p2, p3, p4, p5, p6 = olho_pontos
    
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    
    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Inicializando o MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices oficiais padronizados do MediaPipe para os dois olhos
INDICES_OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
INDICES_OLHO_DIREITO  = [33, 160, 158, 133, 153, 144]


cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # Conversão de BGR para RGB para o MediaPipe processar
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Convertendo de volta para BGR para exibição no OpenCV
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            
            
            pontos_olho_esq = []
            for idx in INDICES_OLHO_ESQUERDO:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                pontos_olho_esq.append((x, y))
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1) # Vermelho para esquerdo

            
            pontos_olho_dir = []
            for idx in INDICES_OLHO_DIREITO:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w) #largura
                y = int(landmark.y * h) #altura
                pontos_olho_dir.append((x, y))
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1) # Azul para direito

            
            ear_esq = calcular_ear(pontos_olho_esq)
            ear_dir = calcular_ear(pontos_olho_dir)
            ear_atual = (ear_esq + ear_dir) / 2.0

            
            cv2.putText(image, f'EAR E: {ear_esq:.2f} | D: {ear_dir:.2f}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(image, f'EAR Media: {ear_atual:.2f}', (20, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            if ear_atual < 0.20:
                cv2.putText(image, 'ALERTA: OLHOS FECHADOS!', (20, 210), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # Renderização da malha facial inteira
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Renderização dos contornos específicos
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Face Mesh - Deteccao de Fadiga', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()