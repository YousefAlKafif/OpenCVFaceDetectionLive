# Importing the libraries
import cv2

# Loading the cascades from OpenCVs GitHub repository. I will use these cascades to detect the face/eyes/smile.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# NOTE: Cascades only work on B&W images so my detection function must use a B&W image.


# FACIAL DETECTION FUNCTION DEFINITION:-
def detect(gray, ReturnFrame): # (Input B&W image , Output colored image w/rectangled features)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Applying detectMultiScale() method from face cascade object to locate all faces in image.
                                                        # Using a 1.3x reduction scale on image (for faster computation), and 5 minimum neighboring zones required(for more accurate detection). 
   
    # GOING THROUGH FACES :- 
    for (x, y, w, h) in faces: # (x,y) = Upper left co-ordinates,  w = width, h = height.
       
        cv2.rectangle(ReturnFrame, (x, y), (x+w, y+h), (255, 0, 0), 2) # BLUE RECTANGLES ON FACES
        # rectangle() arguments -> Image to be 'drawn' on, (upper-left cordinates), (lower-right cordinates), color, thickness.
      
        regionOfInterest_gray = gray[y:y+h, x:x+w] # Establishing region of interest of which the eye & smile cascades will look through. I.e. the B&W image.
        regionOfInterest_color = ReturnFrame[y:y+h, x:x+w] # RoI for the return images. I.e. this RoI will have rectangles on it.
        
        eyes = eye_cascade.detectMultiScale(regionOfInterest_gray, 1.1, 22) # Acquiring eyes locations from the region of interest i.e. the face.
                                                                            # Increased minimum neighbors required to 22.                  
        for (ex, ey, ew, eh) in eyes: # GOING THROUGH EYES :-
            cv2.rectangle(regionOfInterest_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # GREEN RECTANGLES ON EYES.
        
        smiles = smile_cascade.detectMultiScale(regionOfInterest_gray, 1.7, 22) # Acquiring smiles locations from the region of interest i.e. the face.
        for (sx, sy, sw, sh) in smiles: # GOING THROUGH SMILES :-
            cv2.rectangle(regionOfInterest_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2) # RED RECTANGLES ON SMILES.
    
    return ReturnFrame


# WEBCAM CONTROL USING CV2 *for input image & output* :-
video_capture = cv2.VideoCapture(0)

while True: # Will be true infinitely after I execute the code.
    UnwantedReturn, lastframe = video_capture.read() 
    
    gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY) # Converting the last frame into grayscale so that the cascade classifier can be used.
    
    OutputImage = detect(gray, lastframe) # Using my created function to retrieve output image with detection rectangles.
    
    cv2.imshow('Video', OutputImage) # Displaying the output.
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # IF I type 'q' it will break this loop.
        break
    
video_capture.release() # Turns webcam off
cv2.destroyAllWindows() # Destroys all windows that image was displayed on.

