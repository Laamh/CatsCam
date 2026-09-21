'''
ce script à été entièrement généré par IA plus précisément par claude
j'avais besoin d'un outil vite fait pour trier les images
'''

import cv2, os, shutil

SRC = "dataset/a_trier"
DEST = "dataset"

KEYS = {
    ord('a'): "yuki",
    ord('z'): "bom",
    ord('e'): "broutch",
    ord('r'): "titus",
    ord('t'): "toundra",
    ord('x'): None
}

def move_safe(src_path, dest_folder, filename):
    """Déplace le fichier sans jamais écraser un fichier existant du même nom."""
    dest_path = f"{dest_folder}/{filename}"
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(f"{dest_folder}/{name}_{i}{ext}"):
            i += 1
        dest_path = f"{dest_folder}/{name}_{i}{ext}"
    shutil.move(src_path, dest_path)

for name in KEYS.values():
    if name:
        os.makedirs(f"{DEST}/{name}", exist_ok=True)
os.makedirs(f"{DEST}/poubelle", exist_ok=True)

files = [f for f in os.listdir(SRC) if f.endswith(".jpg")]

for f in files:
    path = f"{SRC}/{f}"
    img = cv2.imread(path)
    if img is None:
        continue
    cv2.imshow("Tri - a:yuki z:bom e:broutch r:titus t:toundra x:jeter esc:quitter", img)
    key = cv2.waitKey(0) & 0xFF

    if key == 27:
        break
    elif key in KEYS:
        dest_folder_name = KEYS[key] or "poubelle"
        move_safe(path, f"{DEST}/{dest_folder_name}", f)

cv2.destroyAllWindows()