import cv2
import time
import csv
import os
import sys
import winsound
import importlib.util
from datetime import datetime

from mediapipe.python.solutions import face_mesh as mp_face_mesh


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  


def importar_modulo(nome_modulo, caminho_arquivo):
    """Carrega um .py com hífen no nome como se fosse um módulo normal."""
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho_arquivo)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


caminho_ear = os.path.join(PROJECT_ROOT, "ear-test", "ear-test.py")
caminho_mar = os.path.join(PROJECT_ROOT, "mar-test", "mar-test.py")
caminho_head = os.path.join(PROJECT_ROOT, "head-test", "head-test.py")

ear_mod = importar_modulo("ear_test_mod", caminho_ear)
mar_mod = importar_modulo("mar_test_mod", caminho_mar)
head_mod = importar_modulo("head_test_mod", caminho_head)

calcular_ear = ear_mod.calcular_ear_medio 
calcular_mar = mar_mod.calcular_mar_puro  
calcular_head_pose = head_mod.calcular_head_pose


face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
pTime = 0

gravando = False
rotulo_atual = 0  # 0 = Alerta, 1 = Fadiga
tempo_inicio_bloco = time.time()
duracao_bloco = 3.0

# Cada item: (t_relativo, ear, mar, yaw, pitch) -- um por frame, sem média
bloco_frames = []

# Cada execução do script é uma sessão com seu próprio CSV em dados/.
# Assim PC e notebook não geram conflito no git; depois é só concatenar os arquivos.
id_pessoa = input("Pessoa (ex: eu, mae) [eu]: ").strip() or "eu"
cenario = input("Cenário/iluminação (ex: noite-luz-quarto, dia-janela) [padrao]: ").strip() or "padrao"
id_sessao = datetime.now().strftime("%Y%m%d_%H%M%S")

pasta_dados = os.path.join(BASE_DIR, "dados")
os.makedirs(pasta_dados, exist_ok=True)
arquivo_csv = os.path.join(pasta_dados, f"sessao_{id_sessao}.csv")

id_bloco_atual = 0  # único dentro da sessão; (id_sessao, id_bloco) identifica o bloco no dataset todo

csv_file = open(arquivo_csv, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["id_sessao", "id_pessoa", "cenario", "id_bloco", "frame", "t",
                     "ear", "mar", "yaw", "pitch", "rotulo"])
print(f"[INFO] Sessão {id_sessao} | pessoa: {id_pessoa} | cenário: {cenario}")
print(f"[INFO] Gravando em: {arquivo_csv}")

print("[INFO] Pressione '0' para Alerta, '1' para Fadiga, 's' para pausar e ESC/Q para sair.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    ear_atual, mar_atual, yaw_atual, pitch_atual = 0.0, 0.0, 0.0, 0.0
    rosto_detectado = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            rosto_detectado = True

            # --- Funções reais, importadas de ear-test / mar-test / head-test ---
            ear_atual = calcular_ear(face_landmarks, w, h)
            mar_atual = calcular_mar(face_landmarks, w, h)
            yaw_atual, pitch_atual = calcular_head_pose(face_landmarks, w, h)

    # Lógica de Gravação Automática por Blocos de 3 Segundos com Beep
    if gravando and rosto_detectado:
        tempo_decorrido = time.time() - tempo_inicio_bloco
        bloco_frames.append((tempo_decorrido, ear_atual, mar_atual, yaw_atual, pitch_atual))
        tempo_restante = max(0.0, duracao_bloco - tempo_decorrido)

        cv2.putText(image, f'BLOCO 3S ({rotulo_atual}): {tempo_restante:.1f}s', (20, 150),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 255) if rotulo_atual == 1 else (0, 255, 0), 2)

        # Passados os 3 segundos: gera um ID novo e grava TODOS os frames do bloco (sem média)
        if tempo_decorrido >= duracao_bloco:
            if len(bloco_frames) > 0:
                id_bloco_atual += 1
                for i, (t, e, m, y, p) in enumerate(bloco_frames):
                    csv_writer.writerow([id_sessao, id_pessoa, cenario, id_bloco_atual, i, f"{t:.3f}", f"{e:.4f}", f"{m:.4f}",
                                         f"{y:.4f}", f"{p:.4f}", rotulo_atual])
                csv_file.flush()
                print(f"[SALVO] Bloco #{id_bloco_atual} | Rótulo: {rotulo_atual} | {len(bloco_frames)} frames")

            bloco_frames = []
            tempo_inicio_bloco = time.time()
            winsound.Beep(800, 200) 
    else:
        cv2.putText(image, 'STANDBY (Pressione 0 ou 1)', (20, 150), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 2)

    
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Coleta Mestre Unificada', image)

    
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('0'):  # Ativa modo Alerta
        gravando = True
        rotulo_atual = 0
        bloco_frames = []
        tempo_inicio_bloco = time.time()
        print("[INFO] Modo Alerta ativado (0)")
        winsound.Beep(1200, 150)
    elif key == ord('1'):  # Ativa modo Fadiga
        gravando = True
        rotulo_atual = 1
        bloco_frames = []
        tempo_inicio_bloco = time.time()
        print("[INFO] Modo Fadiga ativado (1)")
        winsound.Beep(500, 300)
    elif key == ord('s'):  # Pausa
        gravando = False
        print("[INFO] Gravação pausada.")

cap.release()
csv_file.close()
cv2.destroyAllWindows()