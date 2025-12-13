# 🚗 Carro Autônomo com Visão Computacional

Projeto acadêmico de um **carro autônomo em ambiente controlado**, utilizando **visão computacional em tempo real** executada em um **Raspberry Pi**.  
O sistema usa um modelo **YOLO convertido para TensorFlow Lite** para detectar elementos da pista e gerar informações para controle do veículo.

---

## 📌 Funcionalidades

- Captura contínua de imagens da câmera
- Inferência em tempo real com YOLO (TFLite)
- Detecção de faixas da pista
- Cálculo de FPS baseado no tempo de inferência
- Desenho de bounding boxes nos frames
- Gravação de trechos de vídeo para análise
- Envio de frames via UDP para monitoramento remoto
- Arquitetura modular e organizada

---

## 🗂 Estrutura do Projeto

projeto/
│
├── main.py # Arquivo principal
├── config.py # Configurações gerais
│
├── camera/
│ └── camera_thread.py # Captura de frames em thread
│
├── inferencia/
│ └── yolo_inference.py # Inferência com YOLO TFLite
│
├── comm/
│ └── udp_sender.py # Envio de imagens via UDP
│
├── modelos/
│ └── yolo-tf.tflite # Modelo treinado
│
├── videos/ # Vídeos gerados para análise
└── README.md


---

## 🧠 Inferência e FPS

O FPS exibido no sistema representa o **FPS real de inferência**, calculado a partir do tempo gasto em:
- pré-processamento
- inferência do modelo
- pós-processamento

Isso garante decisões mais confiáveis para controle do carro.

---

## 🎥 Gravação de Vídeo

Quando uma detecção ocorre, o sistema grava automaticamente um trecho do vídeo com:
- marcações da inferência
- bounding boxes
- métricas exibidas

Os arquivos são salvos na pasta `videos/` para análise posterior.

---

## 📡 Comunicação UDP

O sistema pode enviar frames processados via **UDP**, permitindo:
- monitoramento remoto
- debug sem necessidade de display no Raspberry Pi

---

## ▶️ Como Executar

Ative o ambiente virtual:
```bash
source .venv/bin/activate

Execute o projeto:

python main.py

⚙️ Tecnologias Utilizadas

    Python 3

    OpenCV

    TensorFlow Lite Runtime

    NumPy

    Raspberry Pi OS (Linux)

📍 Observações

    O projeto prioriza tempo real e estabilidade, não FPS máximo

    Resoluções reduzidas são usadas para melhor desempenho

    Código modular facilita manutenção e evolução

🎓 Contexto

Projeto desenvolvido com fins acadêmicos, voltado ao estudo de:

    visão computacional embarcada

    sistemas autônomos

    controle de veículos em tempo real