# Detecção de Sonolência com EAR (Eye Aspect Ratio) + MediaPipe


## 1. Por que usar o EAR?

Em vez de analisar pixels de cor — abordagem que falha com pouca luz ou variações de tom de pele — o EAR usa uma **razão proporcional entre a altura e a largura do olho**.

**Vantagem:** invariância à escala e à distância. Se o usuário se aproximar ou se afastar da câmera, a proporção interna do olho se mantém estável.

### Patamares acadêmicos identificados

| Estado do olho | Valor típico de EAR |
|---|---|
| Olhos abertos | entre **0.30** e **0.45** (varia com a anatomia individual) |
| Olhos fechados / sonolência | cai bruscamente para abaixo de **0.20** (geralmente entre 0.05 e 0.15 nos testes) |

---

## 2. A Inteligência Artificial: MediaPipe Face Mesh

**O que é:** biblioteca e modelo de aprendizado de máquina do Google, otimizada para visão computacional.

**Por que não usar caixas delimitadoras simples?** Porque o projeto precisa de precisão anatômica milimétrica, não de um retângulo genérico ao redor do rosto.

**Funcionamento da malha:** o FaceMesh prediz **468 pontos tridimensionais** (landmarks) da face humana em tempo real, com altíssimo FPS.

### Parâmetros de configuração cruciais

| Parâmetro | Função |
|---|---|
| `max_num_faces=1` | Trava o processamento para analisar apenas um rosto por vez |
| `refine_landmarks=True` | Ativa o refinamento de alta precisão essencial para mapear o contorno exato da íris e das pálpebras |
| `min_detection_confidence=0.5` | Nota de corte de 50% de certeza exigida para a IA "achar" o rosto do zero |
| `min_tracking_confidence=0.5` | Certeza mínima exigida para a IA apenas continuar seguindo o rosto nos frames seguintes, sem precisar reprocessar tudo — o que garante fluidez e alto FPS |

---

## 3. Os índices oficiais do MediaPipe

Indices vem da documentação oficial da google, sao eles abaixo:

### Índices do olho esquerdo

Definidos estritamente na ordem geométrica correta `[P1, P2, P3, P4, P5, P6]` para alimentar a fórmula matemática:

| Ponto | Índice | Descrição |
|---|---|---|
| P1 | 362 | Canto esquerdo do olho |
| P2 | 385 | Pálpebra superior (ponto 1) |
| P3 | 387 | Pálpebra superior (ponto 2) |
| P4 | 263 | Canto direito do olho |
| P5 | 373 | Pálpebra inferior (ponto 2) |
| P6 | 380 | Pálpebra inferior (ponto 1) |

---

## 4. A modelagem matemática (a fórmula do EAR)


![Fórmula do EAR](/photos-explain/ear-formula.png)

**Distâncias verticais** : medem a altura da fenda palpebral em dois pontos diferentes. Usar duas alturas evita falhas caso a piscada seja torta ou o olhar esteja desviado.

**Distância horizontal** : mede o comprimento total do olho, de ponta a ponta.

**O fator 2 no divisor:** normaliza a média das duas alturas verticais em relação à largura total.

**Trava de segurança matemática:** se a distância horizontal for igual a zero (por algum frame corrompido), o código retorna `0.0` para evitar erro crítico de divisão por zero.

### Visualizando a diferença

![Comparação entre olho aberto e olho fechado](/photos-explain/eye-points.png)

Com o olho **aberto**, a distância vertical entre P2↔P6 e P3↔P5 é maior, elevando o valor do EAR. Com o olho **fechado**, essa distância vertical praticamente desaparece, derrubando o EAR — é exatamente essa queda brusca que o algoritmo usa para disparar o alerta de sonolência.

---

## 5. O desafio da conversão de espaço (a matemática dos pixels)

**O formato da IA:** o MediaPipe entrega as coordenadas de forma normalizada, em proporção (valores decimais entre 0.0 e 1.0). Isso garante que o código funcione de forma independente da resolução (720p, 1080p ou 4K).

**A conversão prática (`image.shape`):** para desenhar os pontos na tela, é preciso mapear essa porcentagem para pixels reais usando as dimensões da imagem.

- `image.shape` retorna `(altura, largura, canais_de_cor)`. Usa-se a variável de descarte `_` para jogar fora os canais de cor, guardando a altura em `h` e a largura em `w`.
- **Cálculo base:**
  ```python
  x = int(landmark.x * w)
  y = int(landmark.y * h)
  ```
  Multiplica-se a proporção pela dimensão total da tela e usa-se `int()` para arredondar, já que monitores e matrizes de imagem não trabalham com frações de pixel.

---

## 6. Estratégia de arquitetura e validação do projeto

O projeto é dividido em três códigos, cada um com uma responsabilidade clara:

### Código 1 — Base visual
Validação da captura de vídeo via OpenCV (`cv2.VideoCapture`), conversão obrigatória de cores (BGR → RGB para o MediaPipe processar, e de volta para BGR para a janela do OpenCV exibir), além do desenho da malha (Tesselation e Contours) com cálculo de FPS.

### Código 2 — Análise analítica
Introdução da função matemática `calcular_ear`, varredura dos 6 índices oficiais via loop `for`, armazenamento das coordenadas em lista (`.append()`) e aplicação do limiar de alerta (`ear_atual < 0.20`).

### Código 3 — Validação científica estática
Uso de imagens de controle (`open-eye.jpeg` e `close-eye.jpeg`), configurando o MediaPipe com `static_image_mode=True` e redimensionamento proporcional (`altura_desejada = 720`). Essa etapa comprova a precisão matemática determinística do modelo, validando ~0.31 para olho aberto e ~0.05 para olho fechado, sem depender de ruídos da webcam.

---

## Resumo do pipeline

```
Captura de vídeo (OpenCV)
        │
        ▼
Conversão BGR → RGB
        │
        ▼
MediaPipe Face Mesh (468 landmarks)
        │
        ▼
Extração dos 6 pontos do olho (P1–P6)
        │
        ▼
Cálculo do EAR
        │
        ▼
EAR < 0.20 ?  ──► Sim ──► Alerta de sonolência
        │
        Não
        │
        ▼
Continua monitorando
```
---

## 7. MAR (Mouth Aspect Ratio) - Detecção de Bocejos

Assim como o EAR monitora os olhos, o **MAR** calcula a proporção da abertura da boca para identificar episódios de bocejo excessivo, um dos principais indicadores fisiológicos de sonolência ao volante.

### O Conceito Matemático
A lógica utiliza pontos estratégicos dos lábios superior, inferior e dos cantos da boca para calcular a razão entre a distância vertical e a distância horizontal. 

![Fórmula e Pontos do MAR](/photos-explain/mar-points-formula.png)

* **Distâncias Verticais:** Medem a abertura máxima da boca entre o lábio superior e o inferior.
* **Distância Horizontal:** Mede a largura total da boca (cantos esquerdo e direito).

### A Regra de Negócio (Diferenciando a Fala do Bocejo)
O grande desafio teórico do MAR é que a boca se movimenta constantemente durante a fala, gerando picos rápidos. Para evitar falsos positivos:
* **O Limiar:** Define-se um patamar mínimo (ex: `MAR > 0.50`).
* **Fator Temporal:** O sistema utiliza um contador de frames para garantir que o bocejo seja sustentado por um período contínuo (ex: de 2 a 4 segundos), caracterizando a fadiga profunda em vez de uma simples conversa.

## 8. Head Pose Estimation (Estimativa de Postura da Cabeça)

Além dos olhos (EAR), o projeto monitora a inclinação da cabeça em um espaço tridimensional para detectar o fenômeno do **microssono** (quando o motorista "pesca" a cabeça para frente).

### O Conceito Matemático
Como a câmera captura apenas imagens bidimensionais (2D), o algoritmo utiliza um truque de projeção matemática chamado **solvePnP** (Perspective-n-Point). Ele cruza 6 pontos anatômicos rígidos da face detectados pelo MediaPipe com um **Modelo Antropométrico 3D Padrão** (um crânio humano virtual genérico onde a ponta do nariz é o ponto zero `[0,0,0]`).

![Eixos de Rotação da Cabeça](/photos-explain/head-points.png)

O resultado da matriz de rotação é decomposto nos **Ângulos de Euler**:
* **Pitch (Inclinação Vertical):** Rotação no eixo X. É o foco principal do projeto. Quando o motorista relaxa a musculatura do pescoço e a cabeça cai, o Pitch despenca para valores negativos.
* **Yaw (Rotação Horizontal):** Rotação no eixo Y. Mede se o motorista está olhando para os lados (distração).
* **Roll (Inclinação Lateral):** Rotação no eixo Z. Mede o tombamento da cabeça em direção aos ombros.

### Parâmetros de Alerta
* **Limiar de Pitch (`LIMIAR_PITCH`):** Configurado em **-15°**. Valores inferiores a esse indicam queda crítica da cabeça.
* **Fator Temporal (`FRAMES_NECESSARIOS`):** Exigência de **20 frames consecutivos** (aprox. 1 segundo) abaixo do limiar para disparar o alerta visual, evitando falsos positivos causados por movimentos rápidos ou checagem de painel.