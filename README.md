# Modelo de reconocimiento de gestos en video

Este repositorio contiene todo el pipeline necesario para capturar gestos en video, extraer landmarks 3D con MediaPipe, entrenar un modelo de reconocimiento con TensorFlow y desplegarlo para inferencia en tiempo real desde distintos dispositivos (EC2, Raspberry Pi y un puente para frontends web).

## Tecnologías principales

- **Python 3.9+** como lenguaje base para scripts de captura, entrenamiento e inferencia.
- **TensorFlow** para definir y servir la red LSTM que clasifica secuencias de landmarks.
- **MediaPipe Hands** para detectar las manos y obtener landmarks 3D por cuadro.
- **OpenCV** para capturar video y mostrar retroalimentación visual durante la captura o inferencia local.
- **Flask + Socket.IO** para exponer predicciones vía REST/SSE/websockets al frontend o a clientes remotos.
- **Requests** para que el cliente de Raspberry Pi consulte el servidor EC2.
- **Boto3** para subir datasets y modelos entrenados a buckets S3 cuando se integren credenciales de AWS.

## Estructura del proyecto

```
respaldoModeloHelen/
├── README.md                 # Este documento
├── cliente_raspberry.py      # Cliente que captura video y consulta el servidor EC2
├── server_ec2_flask.py       # API Flask que sirve un SavedModel en una instancia remota
├── frontend_bridge/          # Servicio Flask + Socket.IO para integrarse con un frontend web
│   ├── server.py
│   ├── service.py
│   └── ...
└── video_gesture_model/      # Pipeline completo de captura, extracción y entrenamiento
    ├── capture_videos.py
    ├── extract_landmarks.py
    ├── realtime_inference.py
    ├── train_model.py
    └── ...
```

## Preparación del entorno

1. Crea y activa un entorno virtual opcional.
2. Instala las dependencias base del pipeline:

   ```bash
   pip install -r video_gesture_model/requirements.txt
   ```

3. Para el puente web instala también Flask, Socket.IO y CORS (si no vienen en tu entorno):

   ```bash
   pip install flask flask-socketio flask-cors
   ```

4. Si vas a usar el servidor remoto o la subida a S3, instala `boto3` (incluido en los requirements) y configura tus credenciales con `aws configure` o variables de entorno antes de ejecutar los scripts relacionados.

## Flujo de trabajo del modelo

1. **Captura de clips de entrenamiento**

   ```bash
   python -m video_gesture_model.capture_videos <nombre_gesto>
   ```

   El script graba clips de `config.CLIP_DURATION` segundos por gesto y los guarda en `video_gesture_model/data/raw_videos/<nombre_gesto>` con asistencia en consola para contar muestras.

2. **Extracción de landmarks con MediaPipe**

   ```bash
   python -m video_gesture_model.extract_landmarks [gesto1 gesto2 ...]
   ```

   Obtiene secuencias de landmarks normalizados para ambas manos y genera un `.npz` junto a un `*_labels.json` con el mapeo gesto→índice.

3. **Entrenamiento del modelo LSTM**

   ```bash
   python -m video_gesture_model.train_model \
       --dataset video_gesture_model/data/features/gesture_dataset.npz \
       --labels video_gesture_model/data/features/gesture_dataset_labels.json \
       --epochs 70 --batch-size 24 --learning-rate 7e-4 \
       --lstm-units 160 96 --dense-units 96 --dropout 0.45
   ```

   El script mezcla los datos, reserva un subconjunto para validación y guarda un `SavedModel` con checkpoints y logs listos para TensorBoard en `video_gesture_model/data/models`. Todos los hiperparámetros son configurables mediante flags.

4. **Inferencia en tiempo real (local)**

   ```bash
   python -m video_gesture_model.realtime_inference --model-dir video_gesture_model/data/models/gesture_model_YYYYMMDD_HHMMSS
   ```

   Si no indicas un directorio, el script lista los modelos disponibles para seleccionarlos desde la terminal y muestra la predicción en la ventana de OpenCV.

5. **Carga opcional a AWS S3**

   ```bash
   python -m video_gesture_model.aws_utils dataset --name gesture_dataset
   python -m video_gesture_model.aws_utils model video_gesture_model/data/models/gesture_model_YYYYMMDD_HHMMSS
   ```

   Asegúrate de actualizar `video_gesture_model/config.py` con los nombres reales de tus buckets antes de ejecutar estos comandos.

## Despliegues y consumo del modelo

### Servidor de inferencia en EC2

- Copia el `SavedModel` entrenado y su `labels.json` a la ruta indicada por `MODEL_DIR` en `server_ec2_flask.py`.
- Actualiza `MODEL_DIR`, `LABELS_PATH` y ejecuta el servidor:

  ```bash
  python server_ec2_flask.py
  ```

- El servicio expone `/predict` (POST con `sequence`) y `/health` en el puerto 8000, listos para recibir secuencias desde clientes remotos.

### Cliente Raspberry Pi

- Configura `EC2_URL` con la IP o DNS del servidor.
- Ejecuta el cliente para capturar video, generar la ventana de OpenCV y enviar secuencias al endpoint remoto:

  ```bash
  python cliente_raspberry.py
  ```

- El cliente mantiene un búfer de `SEQUENCE_LENGTH` frames, normaliza los landmarks y muestra la etiqueta más reciente junto con la confianza.

### Puente para frontends web

- Permite reutilizar el modelo desde una Raspberry Pi o PC y transmitir resultados a una UI vía REST, SSE o Socket.IO.
- Inicia el servicio indicando la carpeta del modelo (usa el más reciente por defecto):

  ```bash
  python -m frontend_bridge.server --model-path video_gesture_model/data/models/gesture_model_YYYYMMDD_HHMMSS
  ```

- Consulta `/api/status`, `/api/gestures`, controla la inferencia desde `/api/inference/start|stop` o suscríbete al namespace `/gestures` para recibir eventos `gesture_prediction` en tiempo real.

## Próximos pasos sugeridos

- Ajustar las constantes de `video_gesture_model/config.py` (duración de clips, rutas, buckets S3) a tu entorno antes de iniciar la captura.
- Automatizar la ejecución como servicio (por ejemplo con `systemd`) si vas a usar Raspberry Pi o despliegues permanentes.

Con estos componentes puedes cubrir el ciclo completo: recolectar datos, entrenar el modelo, ejecutarlo localmente o a distancia y exponer sus predicciones a aplicaciones externas.
