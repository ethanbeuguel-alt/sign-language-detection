import os
from PIL import Image



train_folder = "archive/ASL_Dataset/Test"

# Liste uniquement les dossiers
dossiers = [d for d in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, d))]


for d in dossiers:

    # Dossier contenant les images
    input_folder = "archive/ASL_Dataset/Test/" + d
    output_folder = "Dataset_gris/Test/" + d

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Parcours des fichiers
    for filename in os.listdir(input_folder):
        chemin_entree = os.path.join(input_folder, filename)
        chemin_sortie = os.path.join(output_folder, filename)

        # Ouvrir l'image
        image = Image.open(chemin_entree)

        # Convertir en niveau de gris
        image_gris = image.convert("L")

        # Sauvegarder
        image_gris.save(chemin_sortie)

    print(d, " ok")
        #print(f"{filename} convertie en gris")