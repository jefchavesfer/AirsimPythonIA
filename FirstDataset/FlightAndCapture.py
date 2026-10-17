import airsim
import cv2
import numpy as np
import os
import time

def clamp(val):
    """Garante que as coordenadas fiquem estritamente no intervalo [0.0, 1.0]."""
    return max(0.0, min(1.0, val))

# 1. Estrutura de Pastas do Dataset YOLO
OUTPUT_DIR = "dataset_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# 2. Conexão com o AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

VEHICLE_NAME = "BayraktarUAV"
CAMERA_NAME = "ChaseCam"
CLASS_ID = 0  # 0 = Bayraktar TB2

# 3. Inicialização e Controle de Voo
print("Armando os motores do drone...")
client.armDisarm(True, vehicle_name=VEHICLE_NAME)

print("Decolando...")
client.takeoffAsync(timeout_sec=10, vehicle_name=VEHICLE_NAME).join()

print("Iniciando trajetória em velocidade de cruzeiro real (36.1 m/s)...")
cruise_alt = -30.0  # 30 metros de altitude
CRUISE_SPEED = 36.1 # Velocidade real do Bayraktar TB2 (130 km/h)

# Waypoints expandidos para acomodar a maior velocidade
waypoints = [
    airsim.Vector3r(200, 0, cruise_alt),
    airsim.Vector3r(1500, 0, cruise_alt),
    airsim.Vector3r(3000, 1000, cruise_alt),
    airsim.Vector3r(3000, 3000, cruise_alt)
]

# Dispara a navegação assíncrona na velocidade de cruzeiro real
client.moveOnPathAsync(
    path=waypoints,
    velocity=CRUISE_SPEED,
    drivetrain=airsim.DrivetrainType.ForwardOnly,
    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
    vehicle_name=VEHICLE_NAME
)

print("Iniciando captura de alta taxa (30 FPS) em modo EO/IR...")
frame_count = 0

# Equalizador adaptativo para o filtro infravermelho
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

try:
    while True:
        responses = client.simGetImages([
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)
        ], vehicle_name=VEHICLE_NAME)
        
        if not responses or responses[0].width == 0:
            time.sleep(0.01)
            continue

        response = responses[0]
        img_w, img_h = response.width, response.height

        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame_rgb = img_1d.reshape(img_h, img_w, 3)

        # --- PROCESSAMENTO INFRAVERMELHO (EO/IR) ---
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        ir_monochrome = clahe.apply(gray)
        ir_frame = cv2.cvtColor(ir_monochrome, cv2.COLOR_GRAY2BGR)

        # --- COORDENADAS DA BOUNDING BOX ---
        box_w, box_h = 0.18, 0.08
        x_center, y_center = 0.50, 0.15

        yolo_annotation = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

        # --- SALVAMENTO DOS DADOS ---
        frame_name = f"frame_{frame_count:05d}"
        cv2.imwrite(os.path.join(IMAGES_DIR, f"{frame_name}.png"), ir_frame)

        with open(os.path.join(LABELS_DIR, f"{frame_name}.txt"), "w") as f:
            f.write(yolo_annotation)

        # --- PREVIEW COM RETÂNGULO NA TELA ---
        preview_frame = ir_frame.copy()
        x1 = int((x_center - box_w / 2) * img_w)
        y1 = int((y_center - box_h / 2) * img_h)
        x2 = int((x_center + box_w / 2) * img_w)
        y2 = int((y_center + box_h / 2) * img_h)
        
        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview_frame, "EO/IR BAYRAKTAR", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_count += 1
        print(f"Frame IR salvo: {frame_name}.png | Total: {frame_count}")

        cv2.imshow("Sensor Eletrooptico / IR - Preview", preview_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Captura contínua em ~30 FPS
        time.sleep(0.033)

except KeyboardInterrupt:
    print("\nCaptura interrompida pelo usuário.")

finally:
    cv2.destroyAllWindows()
    print(f"\nPipeline concluído com sucesso! Total de frames IR salvos: {frame_count}")