import cv2
import os
from ultralytics import YOLO

# Initialisation du modèle YOLO (téléchargement auto si absent en local)
# model = YOLO("yolo26n.pt")  # n'existe pas, gardé en commentaire par erreur de nommage initial
model = YOLO("yolov8x.pt")  # version "x" = la plus précise, tournera sur GPU (RTX 4070 Super)

# Modèle d'identification individuelle, entraîné sur Bom/Broutch/Titus/Toundra/Yuki
classifier = YOLO("runs/classify/train-3/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.6  # en dessous de ce seuil, on reste sur le label générique cat/dog

# Dossier où seront stockés les crops d'animaux détectés, à trier ensuite manuellement par animal
DATASET_DIR = "dataset/a_trier"
os.makedirs(DATASET_DIR, exist_ok=True)

# Reprend la numérotation là où elle s'était arrêtée, au lieu de repartir de 0 et d'écraser
existing_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
capture_count = max([int(f.replace(".jpg", "")) for f in existing_files], default=0) + 1
frame_counter = 0  # utilisé pour espacer les captures (1 image gardée toutes les 15 détections)

# IP Locale de mon iPad, à vérifier je sais pas combien de temps dure le bail IP et j'ai la flemme de checker sur ma box
video_capture = cv2.VideoCapture("http://192.168.1.99:4747/video?1920x1080")

# Tentative de forcer la résolution/FPS du flux (n'a d'effet que si DroidCam l'autorise en HTTP,
# sinon ces valeurs sont ignorées silencieusement et get() renverra la vraie résolution reçue)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Résolution/FPS réellement obtenus (à vérifier au lancement)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Résolution réelle : {width}x{height}")
print(f"FPS réel : {fps}")

# Boucle principale de lecture du flux vidéo
while True:
    ok, frame = video_capture.read()
    if not ok:
        print("Problème pas de flux / appareil déconnecté")
        break

    # Détection sur la frame courante
    # device=0 fait tourner le modèle sur GPU (nécessite Python 3.12 + torch CUDA, cf. installation)
    results = model(frame, device=0, verbose=False)[0]

    for box in results.boxes:
        cls = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == "person":
            # # Floutage fort (façon pixelisation) des humains détectés, pour la vie privée
            # small = cv2.resize(frame[y1:y2, x1:x2], (10, 10), interpolation=cv2.INTER_LINEAR)
            # frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            continue

        elif cls in ("cat", "dog"):
            crop = frame[y1:y2, x1:x2]
            label = cls  # valeur par défaut (cat/dog générique) si pas assez confiant

            if crop.size > 0:
                # Passage du crop dans le classifieur pour identifier l'animal précis
                cls_result = classifier(crop, verbose=False)[0]
                top1_idx = cls_result.probs.top1
                top1_conf = cls_result.probs.top1conf.item()

                if top1_conf >= CONFIDENCE_THRESHOLD:
                    label = f"{classifier.names[top1_idx]} {top1_conf * 100:.0f}%"

            # Affichage de la boîte de détection + label (nom précis ou cat/dog générique)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Sauvegarde du crop pour enrichir le dataset au fil du temps (optionnel, peut être commenté)
            frame_counter += 1
            if frame_counter % 15 == 0:
                cv2.imwrite(f"{DATASET_DIR}/{capture_count}.jpg", crop)
                capture_count += 1

    # Affichage du flux annoté à l'écran
    cv2.imshow("CatsCam", frame)

    # Sortie de la boucle avec la touche Échap
    if cv2.waitKey(1) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()