import cv2
import math
import os

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

# Índices oficiais padronizados do MediaPipe para os dois olhos
INDICES_OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
INDICES_OLHO_DIREITO  = [33, 160, 158, 133, 153, 144]

def processar_imagem(caminho_img):
    if not os.path.exists(caminho_img):
        print(f"Erro: A imagem '{caminho_img}' não foi encontrada!")
        return

    image = cv2.imread(caminho_img)
    
    # Cálculo proporcional padrão para manter a largura/altura correta ao abrir a foto
    altura_desejada = 720
    h_orig, w_orig, _ = image.shape
    largura_desejada = int(w_orig * (altura_desejada / h_orig))
    image = cv2.resize(image, (largura_desejada, altura_desejada))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                
                
                pontos_olho_esq = []
                for idx in INDICES_OLHO_ESQUERDO:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    pontos_olho_esq.append((x, y))
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1) # Vermelho para esquerdo

                
                pontos_olho_dir = []
                for idx in INDICES_OLHO_DIREITO:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    pontos_olho_dir.append((x, y))
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1) # Azul para direito

                
                ear_esq = calcular_ear(pontos_olho_esq)
                ear_dir = calcular_ear(pontos_olho_dir)
                
                # Média dos dois olhos para um resultado mais estável
                ear_atual = (ear_esq + ear_dir) / 2.0
                
                print(f"[{caminho_img}] -> EAR Esquerdo: {ear_esq:.2f} | EAR Direito: {ear_dir:.2f} | Média: {ear_atual:.2f}")

                
                cv2.putText(image, f'EAR E: {ear_esq:.2f} | D: {ear_dir:.2f}', (30, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(image, f'EAR Media: {ear_atual:.2f}', (30, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                
                if ear_atual < 0.20:
                    cv2.putText(image, 'STATUS: OLHOS FECHADOS', (30, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(image, 'STATUS: OLHOS ABERTOS', (30, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # Renderização da malha facial inteira
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        else:
            print(f"Nenhum rosto encontrado na imagem: {caminho_img}")

    cv2.imshow(f"Teste - {caminho_img}", image)
    print("Pressione qualquer tecla na janela da imagem para ir para a próxima...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

processar_imagem("test-photo/open-eye.jpeg")
processar_imagem("test-photo/close-eye.jpeg")