import cv2
import mediapipe as mp

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abre webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()


# Configura MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Erro: Não conseguiu ler o frame da câmera.")
            break


        # Converte imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Processa imagem
        results = hands.process(image)

        # Volta para BGR para OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Verifica se detectou alguma mão
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os pontos e conexões na mão
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detecta se os dedos estão levantados
                landmarks = hand_landmarks.landmark

                # Lista com status dos dedos (1 = levantado, 0 = dobrado)
                finger_status = []

                # Dedo polegar (compara ponta com base)
                finger_status.append(1 if landmarks[4].x < landmarks[3].x else 0)

                # Dedos indicadores (compara ponta com junta intermediária)
                for tip in [8, 12, 16, 20]:
                    finger_status.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)


                # Traduz para gesto
                if finger_status == [0, 1, 0, 0, 0]:
                    gesture = "Apontando"
                elif finger_status == [0, 1, 1, 0, 0]:
                    gesture = "Paz"
                elif finger_status == [0, 0, 0, 0, 0]:
                    gesture = "Punho fechado"
                elif finger_status == [1, 1, 1, 1, 1]:
                    gesture = "Mão aberta"
                elif finger_status == [1, 0, 0, 0, 0]:
                    gesture = "Joinha"
                elif finger_status == [0, 0, 1, 0, 0]:
                    gesture = "gesto feio"
                elif finger_status == [0, 0, 0, 1, 0]:
                    gesture = "lascado"
                elif finger_status == [0, 0, 0, 0, 1]:
                    gesture = "namorados"
                elif finger_status == [0, 1, 1, 1, 0]:
                    gesture = "numero tres"
                elif finger_status == [0, 0, 1, 1, 1]:
                    gesture = "numero tres"
                elif finger_status == [0, 1, 1, 1, 1]:
                    gesture = "numero quatro"
                else:
                    gesture = "Gesto desconhecido"

                # Mostra o gesto na tela
                cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)

        # Exibe o vídeo
        cv2.imshow('Reconhecimento de Gestos', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
