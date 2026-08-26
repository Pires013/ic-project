import cv2
import numpy as np
import time
from mediapipe.python.solutions import face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# 1. Definindo os pontos 3D genéricos (PADRÃO OPENCV)
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # Ponta do Nariz (Ponto Zero)
    [0.0, 330.0, -65.0],        # Queixo (Y positivo = para baixo)
    [-225.0, -170.0, -135.0],   # Canto olho esquerdo (Y negativo = para cima)
    [225.0, -170.0, -135.0],    # Canto olho direito
    [-150.0, 150.0, -125.0],    # Canto esquerdo da boca
    [150.0, 150.0, -125.0]      # Canto direito da boca
], dtype=np.float64)

# Indices do Media Pipe para os 6 pontos principais
# Ordem: Nariz, Queixo, Olho Esq, Olho Dir, Boca Esq, Boca Dir
INDICES_HEAD = [1, 152, 33, 263, 61, 291]

cap = cv2.VideoCapture(0)

contador_frames_queda = 0
#Definindo os limites para detectar a queda de cabeça e quantidade de frames necessários para disparar o alerta
LIMIAR_PITCH = -15  
FRAMES_NECESSARIOS = 20 

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
            
            
            face_2d = []
            for idx in INDICES_HEAD:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_2d.append([x, y])
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1) 
            
            face_2d = np.array(face_2d, dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
            
            rmat, _ = cv2.Rodrigues(rot_vec)

            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0] # Inclinação Vertical (Sim/Não)
            yaw = angles[1]   # Rotação Horizontal (Olhar pros lados)
            roll = angles[2]  # Inclinação Lateral (Orelha no ombro)

            cv2.putText(image, f'Pitch (Cair Cabeca): {int(pitch)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(image, f'Yaw   (Lado): {int(yaw)}', (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(image, f'Roll  (Lateral): {int(roll)}', (20, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            if pitch < LIMIAR_PITCH:
                contador_frames_queda += 1
            else:
                contador_frames_queda = 0
                
            if contador_frames_queda > FRAMES_NECESSARIOS:
                 cv2.putText(image, 'ALERTA: QUEDA DE CABECA!', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()