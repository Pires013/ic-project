import cv2
import numpy as np
import time
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# Inicializando o Face Mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# 1. Definindo os pontos 3D genéricos de um rosto humano (Molde Padrão)
# Esses valores são aproximados e padronizados na visão computacional
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # Ponta do Nariz
    [0.0, -330.0, -65.0],       # Queixo
    [-225.0, 170.0, -135.0],    # Canto esquerdo do olho esquerdo
    [225.0, 170.0, -135.0],     # Canto direito do olho direito
    [-150.0, -150.0, -125.0],   # Canto esquerdo da boca
    [150.0, -150.0, -125.0]     # Canto direito da boca
], dtype=np.float64)

# Índices equivalentes no MediaPipe para esses mesmos 6 pontos
# Ordem: Nariz, Queixo, Olho Esq, Olho Dir, Boca Esq, Boca Dir
INDICES_HEAD = [1, 152, 33, 263, 61, 291]

cap = cv2.VideoCapture(0)

# Variáveis do nosso contador de queda de cabeça
contador_frames_queda = 0
LIMIAR_PITCH = -15  # Graus de inclinação para baixo (ajuste conforme testar)
FRAMES_NECESSARIOS = 20 # Aprox. 1 segundo com a cabeça baixa

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
            
            # 2. Extraindo os pontos 2D (da tela)
            face_2d = []
            for idx in INDICES_HEAD:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_2d.append([x, y])
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1) # Pontos em Verde
            
            face_2d = np.array(face_2d, dtype=np.float64)

            # 3. Configurando a "Matriz da Câmera" (Foco simulado baseado na resolução)
            focal_length = 1 * w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64) # Assumimos zero distorção de lente

            # 4. A Mágica do solvePnP: Descobrindo a rotação do rosto
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
            
            # Convertendo vetor de rotação para matriz de rotação
            rmat, _ = cv2.Rodrigues(rot_vec)
            
            # Extraindo os Ângulos de Euler (Pitch, Yaw, Roll)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch = angles[0] * 360 # Inclinação Vertical (Sim/Não)
            yaw = angles[1] * 360   # Rotação Horizontal (Olhar pros lados)
            roll = angles[2] * 360  # Inclinação Lateral (Orelha no ombro)

            # Exibindo os valores na tela
            cv2.putText(image, f'Pitch (Cair Cabeca): {int(pitch)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(image, f'Yaw   (Lado): {int(yaw)}', (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            
            # Lógica do Contador de Tempo (Queda de Cabeça)
            # Como abaixar a cabeça gera números negativos menores que -15...
            if pitch < LIMIAR_PITCH:
                contador_frames_queda += 1
            else:
                contador_frames_queda = 0 # Zerou, cabeça voltou ao normal
                
            if contador_frames_queda > FRAMES_NECESSARIOS:
                 cv2.putText(image, 'ALERTA: QUEDA DE CABECA!', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()