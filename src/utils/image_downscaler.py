import os
from PIL import Image


train_folder = "Dataset_gris/Test"


dossiers = [d for d in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, d))]



for d in dossiers:

    # Dossier contenant les images
    input_folder = "Dataset_gris/Test/" + d
    output_folder = "Dataset_gris_reduit/Test/" + d

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Parcours des fichiers
    for filename in os.listdir(input_folder):
        chemin_entree = os.path.join(input_folder, filename)
        chemin_sortie = os.path.join(output_folder, filename)


        image = Image.open(chemin_entree)

        # Redimensionnement
        image = image.resize((128, 128))

        image.save(chemin_sortie)
    
    print(d + "ok")

