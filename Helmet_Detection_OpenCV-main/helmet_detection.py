from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO("helmet.pt")

# Videoyu aç
cap = cv2.VideoCapture("helmet2.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Modeli kullanarak kareyi analiz et
    results = model(frame)

    # Sonuçları göster
    annotated_frame = results[0].plot()
    
    resized_frame = cv2.resize(annotated_frame, (0, 0), fx=0.5, fy=0.5)
    
    cv2.imshow("Helmet Detection", resized_frame)

    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
