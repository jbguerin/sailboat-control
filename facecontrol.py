from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils, dlib, cv2, math, socket


# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    B = dist.euclidean(mouth[3], mouth[-1])
    mar = B / A
    return mar


EYE_AR_THRESH = 0.26        # Constante en-deça de laquelle l'oeil est considéré comme fermé
EYE_AR_CONSEC_FRAMES = 7    # Nombre de frames minimum pour considérer l'oeil comme fermé
MOUTH_AR_THRESH = 0.5       # Constante au dela de laquelle la bouche est considérée comme ouverte
MOUTH_AR_CONSEC_FRAMES = 7  # Nombre de frames minimum pour considérer la bouche comme ouverte

EYE_CLOSE_FRAMES = 10   # Nombre de frames pendant lesquelles l'oeil est encore considéré comme fermé
OPEN_COUNTER_RIGHT = 0  # Compteur oeil droit
OPEN_COUNTER_LEFT = 0   # Compteur oeil gauche
COUNTER_RIGHT = 0
COUNTER_LEFT = 0
OEIL = False            # Oeil=False s'il est ouvert, et True s'il est fermé


#Acquisition du flux vidéo
cap = cv2.VideoCapture(0)

# face_detector de dlib
detector = dlib.get_frontal_face_detector()
# base de données de visages
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Coordonnées des yeux
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # oeil gauche
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # oeil droit

compteur_frame = 0


hote = "localhost"
port = 12800
connexion_avec_serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connexion_avec_serveur.connect((hote, port))
print("Connexion établie avec le serveur sur le port {}".format(port))

# Message à envoyer au serveur
commande_a_envoyer = "0"

while True:
    compteur_frame += 1
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tickmark = cv2.getTickCount()
    faces = detector(gray, 0)
    if faces is not None:
        # Si pas de visage détécté, Matrice de zéros
        i = np.zeros(shape=(frame.shape), dtype=np.uint8)
    for face in faces:
        landmarks = predictor(gray, face)

        # Calcul de coefficients pour l'inclinaison de la tête.
        d_eyes = math.sqrt(math.pow(landmarks.part(36).x - landmarks.part(45).x, 2) + math.pow(
            landmarks.part(36).y - landmarks.part(45).y, 2))
        d1 = math.sqrt(math.pow(landmarks.part(36).x - landmarks.part(30).x, 2) + math.pow(
            landmarks.part(36).y - landmarks.part(30).y, 2))
        d2 = math.sqrt(math.pow(landmarks.part(45).x - landmarks.part(30).x, 2) + math.pow(
            landmarks.part(45).y - landmarks.part(30).y, 2))
        coeff = d1 + d2
        # Calcul de coefficients pour l'inclinaison de la tête.
        a1 = int(250 * (landmarks.part(36).y - landmarks.part(45).y) / coeff)
        a2 = int(250 * (d1 - d2) / coeff)
        cosb = min((math.pow(d2, 2) - math.pow(d1, 2) + math.pow(d_eyes, 2)) / (2 * d2 * d_eyes), 1)
        a3 = int(250 * (d2 * math.sin(math.acos(cosb)) - coeff / 4) / coeff)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            if n == 30 or n == 36 or n == 45:
                cv2.circle(i, (x, y), 3, (255, 255, 0), -1)
            else:
                cv2.circle(i, (x, y), 3, (255, 0, 0), -1)

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[48:58]    # Coordonées de la bouche
        # Marque les contours de la bouche
        cv2.circle(frame, (mouth[0][0], mouth[0][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (mouth[-1][0], mouth[-1][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (mouth[3][0], mouth[3][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (mouth[6][0], mouth[6][1]), 3, (0, 0, 255), -1)
        # Calcul du mouth aspect ratio
        mar = mouth_aspect_ratio(mouth)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        ##############################################################
        ######################## OEIL GAUCHE #########################
        ##############################################################
        if leftEAR < EYE_AR_THRESH:
            # Si l'oeil gauche est considéré comme fermé
            COUNTER_LEFT += 1  # Incrémentation du compteur de frame
            OPEN_COUNTER_LEFT = 0
            cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                # Si l'oeil est fermé depuis "EYE_AR_CONSEC_FRAMES", afficher qu'il est fermé
                OEIL = True  # Oeil fermé
        else:
            OPEN_COUNTER_LEFT += 1
            if OEIL == True and OPEN_COUNTER_LEFT < EYE_CLOSE_FRAMES:
                # Si l'oeil était fermé et qu'il n'est ouvert que depuis EYE_CLOSE_FRAMES frames,
                # le considérer encore comme fermé
                cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                OEIL = True

            else:
                # Sinon, le considérer comme ouvert
                cv2.putText(frame, "EAR: {:.2f}".format(leftEAR), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER_LEFT = 0
                OEIL = False


        ##############################################################
        ######################### OEIL DROIT #########################
        ##############################################################
        if rightEAR < EYE_AR_THRESH:
            # Si l'oeil droit est considéré comme fermé
            COUNTER_RIGHT += 1
            OPEN_COUNTER_RIGHT = 0
            cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                OEIL = True
        else:
            OPEN_COUNTER_RIGHT += 1
            if OEIL == True and OPEN_COUNTER_RIGHT < EYE_CLOSE_FRAMES:
                cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                OEIL = True
            else:
                cv2.putText(frame, "EAR: {:.2f}".format(rightEAR), (300, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER_RIGHT = 0
                OEIL = False


        ##############################################################
        ########################## ACTION ############################
        ##############################################################
        if OEIL == True and mar > MOUTH_AR_THRESH and a1 < 40 and a1 > -40:
            # Si l'oei est fermé et la bouche ouverte
            commande_a_envoyer = str(3)
        else:
            if a1 < -40 and OEIL == True:
                # Si la tête est inclinée vers la gauche et au moins un oeil est fermé
                commande_a_envoyer = str(1)
            elif a1 > 40 and OEIL == True:
                # Si la tête est inclinée vers la droite et au moins un oeil est fermé
                commande_a_envoyer = str(2)
            else:
                # Si la tête n'est pas inclinée
                commande_a_envoyer = str(0)


    # Communication avec le serveur
    connexion_avec_serveur.send(str(commande_a_envoyer).encode('utf8'))

    # Visualisation de l'utilisateur
    cv2.imshow("Utilisateur", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        connexion_avec_serveur.send(("fin").encode())
        break

cap.release()
cv2.destroyAllWindows()
