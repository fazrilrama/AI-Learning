import cv2
import os

# Mendapatkan path absolut dari file ini
current_dir = os.path.dirname(__file__)

# Menyusun path absolut untuk deploy_age.prototxt dan age_net.caffemodel
deploy_path = os.path.join(current_dir, 'deploy_age.prototxt')
model_path = os.path.join(current_dir, 'age_net.caffemodel')

# Load pre-trained Haar Cascade classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained age estimation model dari OpenCV
age_net = cv2.dnn.readNetFromCaffe(deploy_path, model_path)

# Open a video capture object (0 adalah indeks kamera default)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lakukan deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]

        # Prepare input blob for age prediction
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Set input to age network and perform inference
        age_net.setInput(blob)
        age_preds = age_net.forward()

        # Get predicted age
        age = age_preds[0].dot(range(0, 101))

        # Draw rectangle around the face and display age
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Age: {int(age)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Tampilkan frame dengan informasi umur
    cv2.imshow('Age Estimation', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ketika semuanya selesai, lepaskan objek video capture dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
