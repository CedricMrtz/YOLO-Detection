from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\cedri\OneDrive - Instituto Educativo del Noroeste, A.C\SeaFox\YOLO\best.pt")



# Perform object detection on an image
results = model(r"C:\Users\cedri\OneDrive - Instituto Educativo del Noroeste, A.C\SeaFox\YOLO\Captura de pantalla 2025-01-10 141359.png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model