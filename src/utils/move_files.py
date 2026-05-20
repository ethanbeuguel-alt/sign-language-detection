import os
import shutil
import random

def deplacer_images(source_folder, destination_folder, n):

    # Lister les fichiers image dans le dossier source
    images = [f for f in os.listdir(source_folder)]

    
    # Choisir n images aléatoirement
    images_a_deplacer = random.sample(images, n)
    

    os.makedirs(destination_folder, exist_ok=True)

    # Déplacer les images
    for img in images_a_deplacer:
        src_path = os.path.join(source_folder, img)
        dst_path = os.path.join(destination_folder, img)
        shutil.move(src_path, dst_path)
        print(f"Déplacé : {img}")

def copier_images(source_folder, destination_folder, n):

    # Lister les fichiers image dans le dossier source
    images = [f for f in os.listdir(source_folder)]

    
    # Choisir n images aléatoirement
    images_a_deplacer = random.sample(images, n)
    

    os.makedirs(destination_folder, exist_ok=True)

    # Déplacer les images
    for img in images_a_deplacer:
        src_path = os.path.join(source_folder, img)
        dst_path = os.path.join(destination_folder, img)
        shutil.copy(src_path, dst_path)
        print(f"Copié : {img}")

def renommer_images(source_folder, tag):
    for img in os.listdir(source_folder):
        os.rename(source_folder + img, source_folder + tag + "_" + img)


source_folder = "Dataset1/Train/"
dest_folder = "DS_mix/Train/"

dossiers = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

"""
for d in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] :
    source = source_folder + d
    renommer_images(source+ "/", "2")
    #destination = dest_folder + d
    #copier_images(source, destination, 3000)
    print(d, " OK")
"""


