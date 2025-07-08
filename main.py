#python attendance_system.py 
#pip install face-recognition ultralytics numpy
import cv2 #cv2: bibliothèque OpenCV pour le traitement d’images et l’accès à la webcam.
import numpy as np #numpy: pour les tableaux et calculs numériques.
import face_recognition # pour la reconnaissance faciale
import os # pour gérer les fichiers/dossiers.
from datetime import datetime  #pour obtenir la date/heure actuelle.
from ultralytics import YOLO # pour utiliser le modèle YOLOv8 pour la détection d’objets.

class PresenceSystem: #Constructeur qui initialise tout : modèle, visages connus, webcam, etc.
    def __init__(self):
        # Charger YOLOv8n pour la détection générale
        self.model = YOLO('yolov8n.pt')
       
        # Charger les visages connus
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces("known_faces")
        
        # Liste complète des élèves
        self.liste_complete = self.load_student_list("liste_eleves.txt")
        self.presents = set()
        self.absents = set(self.liste_complete)
        
        # Démarrer la webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Erreur: Impossible d'ouvrir la webcam")

    def load_student_list(self, filename):
        """Charge la liste complète des élèves"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Création avec des exemples...")
            return []

    def load_known_faces(self, known_faces_dir):
        """Charge les visages de référence"""
        if not os.path.exists(known_faces_dir):
            print(f"Dossier {known_faces_dir} non trouvé. Création...")
            os.makedirs(known_faces_dir)
            return
            
        for filename in os.listdir(known_faces_dir):#Parcourt le dossier known_faces.
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        # Extraire le nom du fichier
                        name = os.path.splitext(filename)[0].replace("_", " ")
                        self.known_face_names.append(name)
                        print(f"Visage chargé: {name}")
                    else:
                        print(f"Aucun visage trouvé dans {filename}")
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {e}")

    def process_frame(self, frame):
        """Traite une frame pour détecter les visages"""
        # Redimensionner pour améliorer les performances
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Détection des visages
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparaison avec les visages connus
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            name = "Inconnu"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                self.presents.add(name)
                if name in self.absents:
                    self.absents.remove(name)

            # Ajuster les coordonnées pour l'affichage
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Dessiner le rectangle et le nom
            color = (0, 255, 0) if name != "Inconnu" else (255, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Fond pour le texte
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def run(self):
        """Boucle principale du système"""
        try:
            print("Système de présence démarré. Appuyez sur 'q' pour quitter.")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erreur lors de la lecture de la frame")
                    break

                # Détection des visages
                frame = self.process_frame(frame)

                # Détection générale avec YOLO
                results = self.model(frame, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            if box.conf is not None and float(box.conf) > 0.5:
                                class_id = int(box.cls)
                                if class_id < len(self.model.names) and self.model.names[class_id] == "person":
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, "Personne", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Afficher les statistiques
                self.display_stats(frame)

                # Afficher la frame
                cv2.imshow('Systeme de Presence', frame)

                # Vérifier si l'utilisateur veut quitter
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Sauvegarder manuellement
                    self.save_attendance()
                    print("Rapport de présence sauvegardé!")

        except Exception as e:
            print(f"Erreur: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_attendance()

    def display_stats(self, frame):
        """Affiche les statistiques de présence"""
        # Fond semi-transparent pour les stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Statistiques
        stats = f"Presents: {len(self.presents)}/{len(self.liste_complete)}"
        cv2.putText(frame, stats, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Afficher les premiers absents
        if len(self.absents) > 0:
            absents_list = sorted(list(self.absents))[:3]  # Afficher max 3 noms
            abs_text = "Absents: " + ", ".join(absents_list)
            if len(self.absents) > 3:
                abs_text += f" (+{len(self.absents) - 3} autres)"
            cv2.putText(frame, abs_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def save_attendance(self):
        """Sauvegarde le rapport de présence"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"presence_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Rapport de présence - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total élèves: {len(self.liste_complete)}\n")
            f.write(f"Présents: {len(self.presents)}\n")
            f.write(f"Absents: {len(self.absents)}\n")
            f.write(f"Taux de présence: {len(self.presents)/len(self.liste_complete)*100:.1f}%\n\n")
            
            f.write("Élèves présents:\n")
            f.write("-" * 30 + "\n")
            for student in sorted(self.presents):
                f.write(f"✓ {student}\n")
            
            f.write("\nÉlèves absents:\n")
            f.write("-" * 30 + "\n")
            for student in sorted(self.absents):
                f.write(f"✗ {student}\n")
        
        print(f"Rapport sauvegardé: {filename}")


if __name__ == "__main__":
    # Vérifier que le dossier des visages connus existe
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("Dossier 'known_faces' créé. Ajoutez-y des photos nommées 'prenom_nom.jpg'")
        print("Exemple: jean_dupont.jpg, marie_martin.jpg")

    # Vérifier la liste des élèves
    if not os.path.exists("liste_eleves.txt"):
        with open("liste_eleves.txt", 'w', encoding='utf-8') as f:
            f.write("Jean Dupont\nMarie Martin\nPierre Durand\nSophie Leblanc\nLuc Bernard")
        print("Fichier 'liste_eleves.txt' créé avec des exemples. Modifiez-le avec vos élèves.")

    try:
        system = PresenceSystem()
        system.run()
    except Exception as e:
        print(f"Erreur lors du démarrage du système: {e}")



















