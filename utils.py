import cv2
import numpy as np
from keras.models import load_model

def _draw(frame, boxes, probs):
        '''
        Draw bounding box, probs, and landmarks
        '''
        for box, prob, in zip(boxes, probs):
            # draw rectangle on frame
            cv2.rectangle(frame, (box[0], box[1]),
                                 (box[2], box[3]),
                                 (0,0,255), thickness=2)
            # show probability
            cv2.putText(frame, str(prob), (box[2], box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,
                        cv2.LINE_AA)
        return frame

def _detect_rois(boxes):
        '''
        return rois as a list
        '''
        rois = list()
        for box in boxes:
            roi = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            rois.append(roi)
        return rois

def emotion_class(face):
        # model & labels
        emotion_classifier = load_model('trained_model\\fer2013_mini_XCEPTION.107-0.66.hdf5')
        emotions = ['Angry', 'Disgusting', 'Fearful', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # convert color to gray scale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # resize the image to 64x64 for neural network
        f = cv2.resize(gray, (64,64))
        f = f.astype('float')/255.0
        f = np.array(f)
        f = np.expand_dims(f, axis=0)
        f = f[:,:,:,np.newaxis]

        # emotion predict
        preds = emotion_classifier.predict(f)[0]
        emotion_probability = np.max(preds)
        label = emotions[preds.argmax()]
        return preds, label

def gender_class(face):
        # model & labels
        gender_classifier = load_model('trained_model\\gender_mini_XCEPTION.21-0.95.hdf5')
        gender = ['man', 'woman']

        # convert color to gray scale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # resize the image to 64x64 for neural network
        f = cv2.resize(gray, (64,64))
        f = f.astype('float')/255.0
        f = np.array(f)
        f = np.expand_dims(f, axis=0)
        f = f[:,:,:,np.newaxis]

        # gender predict
        preds = gender_classifier.predict(f)[0]
        gender_probability = np.max(preds)
        label = gender[preds.argmax()]
        return label