import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

file_name = 0

webcam=cv2.VideoCapture(file_name)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    i,im=webcam.read()
    if not i :
        break
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)

    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            emotion_prob = np.max(pred)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            #cv2.putText(im,prediction_label)
            '''cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)'''
            confidence_threshold = 0.4
            if emotion_prob > confidence_threshold:
                cv2.putText(im, f'{prediction_label} ({emotion_prob:.2f})', (p - 10, q - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            else:
                cv2.putText(im, 'Unknown', (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("Output", im)

        # Display the frame rate
        fps = webcam.get(cv2.CAP_PROP_FPS)
        print(f"Frame Rate: {fps:.2f}")

        key = cv2.waitKey(27) & 0xFF
        if key == ord('q'):
            break

    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()