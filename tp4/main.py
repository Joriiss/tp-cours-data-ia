import cv2
import numpy as np
import sys
import os


def convert_to_black_white(image_path, output_path=None):
    """
    Convertit une image en noir et blanc (niveaux de gris).
    
    Args:
        image_path: Chemin vers l'image à convertir
        output_path: Chemin de sortie (optionnel, généré automatiquement si non fourni)
    
    Returns:
        Chemin de l'image sauvegardée
    """
    # Charger l'image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Convertir en niveaux de gris (noir et blanc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Générer le chemin de sortie si non fourni
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_bw{extension}"
    
    # Sauvegarder l'image en noir et blanc
    cv2.imwrite(output_path, gray)
    
    return output_path


def load_colorization_model(model_dir="./models"):
    """
    Charge le modèle de colorisation CNN pré-entraîné.
    
    Args:
        model_dir: Répertoire contenant les fichiers du modèle
    
    Returns:
        Tuple (net, pts) - Le réseau et les points de quantification
    """
    prototxt = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    points_file = os.path.join(model_dir, "pts_in_hull.npy")
    model_file = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    
    # Vérifier que les fichiers existent
    if not os.path.exists(prototxt):
        raise FileNotFoundError(f"Fichier prototxt non trouvé: {prototxt}")
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"Fichier points non trouvé: {points_file}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Fichier modèle non trouvé: {model_file}")
    
    print("Chargement du modèle de colorisation...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model_file)
    pts = np.load(points_file)
    
    # Préparer les couches du modèle
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net, pts


def colorize_with_cnn(image_path, model_dir="./models", output_path=None, display=False):
    """
    Colorise une image en N&B en utilisant un modèle CNN pré-entraîné.
    
    Args:
        image_path: Chemin vers l'image en N&B
        model_dir: Répertoire contenant les fichiers du modèle
        output_path: Chemin de sortie (optionnel)
        display: Si True, affiche les images (originale et colorisée)
    
    Returns:
        Chemin de l'image colorisée
    """
    # Charger le modèle
    net, _ = load_colorization_model(model_dir)
    
    # Charger et prétraiter l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Normaliser l'image
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Redimensionner pour le réseau
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Effectuer la colorisation
    print("Colorisation de l'image...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Redimensionner les canaux a et b à la taille originale
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # Combiner les canaux et convertir en BGR
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    # Générer le chemin de sortie
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_colorized{extension}"
    
    # Sauvegarder
    cv2.imwrite(output_path, colorized)
    
    # Afficher si demandé
    if display:
        cv2.imshow("Original", image)
        cv2.imshow("Colorized", colorized)
        print("Appuyez sur une touche pour fermer les fenêtres...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return output_path


def main():
    """Fonction principale."""
    if len(sys.argv) < 3:
        print("Usage: python main.py <mode> <chemin_image> [options]")
        print("\nModes:")
        print("  bw       - Convertir une image couleur en noir et blanc")
        print("  colorize - Coloriser une image N&B avec un modèle CNN pré-entraîné")
        print("\nOptions pour 'colorize':")
        print("  --model-dir <dir>   - Répertoire contenant les fichiers du modèle (défaut: ./models)")
        print("  --display           - Afficher les images (originale et colorisée)")
        print("\nExemples:")
        print("  python main.py bw image.jpg")
        print("  python main.py colorize image_bw.jpg")
        print("  python main.py colorize image_bw.jpg --model-dir ./models --display")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == 'bw':
        if len(sys.argv) < 3:
            print("Erreur: chemin d'image manquant")
            sys.exit(1)
        
        image_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        try:
            result_path = convert_to_black_white(image_path, output_path)
            print(f"Image convertie en noir et blanc: {result_path}")
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif mode == 'colorize':
        if len(sys.argv) < 3:
            print("Erreur: chemin d'image manquant")
            sys.exit(1)
        
        image_path = sys.argv[2]
        model_dir = "./models"
        output_path = None
        display = False
        
        # Parser les options
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--model-dir' and i + 1 < len(sys.argv):
                model_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--display':
                display = True
                i += 1
            elif not sys.argv[i].startswith('--'):
                output_path = sys.argv[i]
                i += 1
            else:
                i += 1
        
        try:
            result_path = colorize_with_cnn(image_path, model_dir, output_path, display)
            print(f"Image colorisée sauvegardée: {result_path}")
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print(f"Mode invalide: {mode}")
        print("Utilisez 'bw' pour convertir en noir et blanc ou 'colorize' pour coloriser")
        sys.exit(1)


if __name__ == '__main__':
    main()
