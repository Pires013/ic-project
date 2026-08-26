import cv2
import time
import math

from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing

def calcular_mar(boca_pontos):
    """
    Calcula o Mouth Aspect Ratio (MAR) com base em 6 pontos da boca.
    """
    p1, p2, p3, p4, p5, p6 = boca_pontos
    
    vertical_1 = math.dist(p3, p4)
    vertical_2 = math.dist(p5, p6)
    horizontal = math.dist(p1, p2)
    
    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return mar

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True)

# Índices oficiais da boca no MediaPipe
INDICES_BOCA = [61, 291, 37, 84, 267, 314]

cap = cv2.VideoCapture(0)

contador_frames_bocejo = 0
LIMIAR_MAR = 0.50 
FRAMES_NECESSARIOS = 30 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            pontos_boca = []
            for idx in INDICES_BOCA:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                pontos_boca.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 255, 0), -1) 

            mar_atual = calcular_mar(pontos_boca)
            
            cv2.putText(image, f'MAR: {mar_atual:.2f}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            if mar_atual > LIMIAR_MAR:
                contador_frames_bocejo += 1
            else:
                contador_frames_bocejo = 0 

            if contador_frames_bocejo > FRAMES_NECESSARIOS:
                cv2.putText(image, 'BOCEJO DETECTADO!', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow('Deteccao de Bocejo (MAR)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()