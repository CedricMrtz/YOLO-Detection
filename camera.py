from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO
model = YOLO(r'C:\Users\cedri\OneDrive - Instituto Educativo del Noroeste, A.C\SeaFox\YOLO\best.pt')  # Puedes usar el modelo preentrenado o uno personalizado

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada, o especifica otro índice

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # Realizar la detección
    results = model(frame)

    # Dibujar los resultados sobre la imagen
    annotated_frame = results[0].plot()  # Devuelve el frame con las anotaciones

    # Mostrar la imagen
    cv2.imshow("Detección en Vivo", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
