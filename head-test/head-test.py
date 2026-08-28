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

# Ordem: Nariz, Queixo, Olho Esq, Olho Dir, Boca Esq, Boca Dir
# Indices do Media Pipe para os 6 pontos principais
INDICES_HEAD = [1, 152, 33, 263, 61, 291]
INDICE_QUEIXO = 152  # Índice exato do queixo no MediaPipe para monitorar a altura física em pixels

cap = cv2.VideoCapture(0)

contador_frames_queda = 0
FRAMES_NECESSARIOS = 50 

# Variáveis de Calibração Dinâmica (Pitch e Altura do Queixo)
calibrado_head = False
valores_pitch_calibracao = []
valores_queixo_calibracao = []
FRAMES_CALIBRACAO = 90  # Aprox. 3 segundos olhando reto
LIMIAR_PITCH = 0.0      
LIMIAR_QUEIXO_Y = 0.0   # Limite alternativo baseado em pixels da tela

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
            
            # Posição Y atual do queixo em pixels na tela
            queixo_y_atual = int(face_landmarks.landmark[INDICE_QUEIXO].y * h)

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

            pitch = angles[0] # Inclinação Vertical
            yaw = angles[1]   # Rotação Horizontal
            roll = angles[2]  # Inclinação Lateral

            # --- FASE DE CALIBRAÇÃO DINÂMICA COMPLETA ---
            if not calibrado_head:
                valores_pitch_calibracao.append(pitch)
                valores_queixo_calibracao.append(queixo_y_atual)
                cv2.putText(image, 'CALIBRANDO HEAD... Olhe reto para a frente', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 255), 2)
                
                if len(valores_pitch_calibracao) >= FRAMES_CALIBRACAO:
                    pitch_neutro = sum(valores_pitch_calibracao) / len(valores_pitch_calibracao)
                    queixo_neutro_y = sum(valores_queixo_calibracao) / len(valores_queixo_calibracao)
                    
                    # Limiar de pitch abaixo da postura neutra
                    LIMIAR_PITCH = pitch_neutro - 18.0
                    # Limiar de queixo: se o queixo descer mais de 45 pixels para baixo na tela
                    LIMIAR_QUEIXO_Y = queixo_neutro_y + 45.0 
                    
                    calibrado_head = True
                    print(f"Calibração concluída! Pitch Neutro: {pitch_neutro:.2f} | Queixo Neutro Y: {queixo_neutro_y:.2f}")
                continue
            # --------------------------------------------

            cv2.putText(image, f'Pitch: {int(pitch)} (Lim: {int(LIMIAR_PITCH)})', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(image, f'Yaw   (Lado): {int(yaw)}', (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(image, f'Roll  (Lateral): {int(roll)}', (20, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # CONDIÇÃO DUPLA DE QUEDA: 
            # Dispara se o ângulo de pitch cair OU se o queixo descer muito verticalmente na tela (salvando do Gimbal Lock)
            cabeca_caindo = (pitch < LIMIAR_PITCH) or (queixo_y_atual > LIMIAR_QUEIXO_Y)

            if cabeca_caindo:
                contador_frames_queda += 1
            else:
                contador_frames_queda = 0
                
            if contador_frames_queda > FRAMES_NECESSARIOS:
                 cv2.putText(image, 'ALERTA: QUEDA DE CABECA!', (20, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()