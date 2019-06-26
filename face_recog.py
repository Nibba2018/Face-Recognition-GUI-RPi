# Import necessary packages
from PyQt5 import QtCore, QtGui, QtWidgets
from name_dialog import Ui_Dialog
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import numpy as np
from PIL import Image

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize and start the video frame capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

labels = {}

def createDict():
    path = './dataset'
    fileList = os.listdir(path)
    #print(fileList)
    for a in fileList:
        b = a.split('.')
        x = b[1]
        if x in labels:
            labels[x] = b[0]
        elif x.isdigit():
            labels[x] = b[0]
            
    print(labels)
    

def fRecogniser(UI):
    UI.label.setText("Looking for Faces....")
    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load the trained mode
    recognizer.read('trainer/trainer.yml')

    # Set the font style
    font = cv2.FONT_HERSHEY_SIMPLEX

    #labels = {'1': "person name"}


    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
        #Read Video Frame
        image = frame.array
        
        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=5)
        
        for (x,y,w,h) in faces:
            
            # Create rectangle around the face
            cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            
            # Recognize the face belongs to which ID(returns a tupple)
            Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
            
            # Check the ID if exist
            print(Id, conf) 
            if conf >30 and conf < 90:
                name = labels[str(Id)]
            else:
                name = "Unknown"
                
            # Put text describe who is in the picture
            cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(image, str(name), (x,y-40), font, 2, (255,255,255), 3)
        
        #show the processed feed    
        cv2.imshow('Looking For Faces.....', image)
        
        #Truncate feed to proper format
        rawCapture.truncate(0)
        
        #Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            UI.label.setText("Raspberry-Pi Face Recognition")
            break

    # Close all windows
    cv2.destroyAllWindows()


def nFaces():
    createDict()
    return len(labels)

def getImagesAndLabels(path,UI):

    progress = 0
    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path

    UI.label.setStyleSheet('color: purple')
    UI.label.setText("Learning Faces......")

    for imagePath in imagePaths:

        UI.progressBar.setValue(progress)
        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)

        # Get the face from the training images
        faces = face_detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)
        progress += 0.5
        UI.progressBar.setValue(progress)

    # Pass the face array and IDs array
    UI.progressBar.setValue(100)
    return faceSamples,ids

def datasetFace(UI):
    face_id = UI.faces + 1

    count = 0

    UI.label.setText("Reading Faces.....")

    #Grabs frames from PiCam
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
        #Capture Video frame as an numpy array
        image = frame.array
        
        #Convert Image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Detect Faces
        faces = face_detector.detectMultiScale(image, scaleFactor = 1.05, minNeighbors=5)
        
        for (x,y,w,h) in faces:
            #Crop the face
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            
            #Increase sample count
            count += 1
            UI.progressBar.setValue(count)
            #save cropped faces
            cv2.imwrite("dataset/" + UI.face_name + '.' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        
        #Display the caped image    
        cv2.imshow('Reading Face....', image)
        
        #Truncate video stream for indivisual frames
        rawCapture.truncate(0)
        #Press Q to quit taking pictures
        if cv2.waitKey(1) & 0xFF == ord('q'):
            UI.progressBar.setValue(0)
            break
        #Automatically stops when 100 pictures are taken
        elif count>100:
            UI.progressBar.setValue(0)
            break
    
    UI.label.setText("Face Data Saved.")
    # Close all started windows
    cv2.destroyAllWindows()




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(709, 307)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.new_face = QtWidgets.QPushButton(self.centralwidget)
        self.new_face.setGeometry(QtCore.QRect(40, 70, 201, 23))
        self.new_face.setObjectName("new_face")
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(270, 70, 181, 23))
        self.train.setObjectName("train")
        self.recog_face = QtWidgets.QPushButton(self.centralwidget)
        self.recog_face.setGeometry(QtCore.QRect(480, 70, 171, 23))
        self.recog_face.setObjectName("recog_face")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(50, 190, 611, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 130, 611, 31))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 250, 241, 21))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.faces = nFaces()
        self.new_face.clicked.connect(self.addFace)
        self.train.clicked.connect(self.training)
        self.recog_face.clicked.connect(self.recognise)
        self.label_2.setText("No. of Faces Saved:" + str(self.faces))
    
    def addFace(self):
        self.progressBar.setValue(0)
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        rsp = Dialog.exec_()
        if rsp == QtWidgets.QDialog.Accepted:
            self.face_name = ui.plainTextEdit.toPlainText()
            datasetFace(self)
        
        self.faces = nFaces()
        self.label_2.setText("No. of Faces Saved:" + str(self.faces))

    def training(self):
        self.progressBar.setValue(0)
        
        #creates Face Disctionaries
        createDict()
        
        # Create Local Binary Patterns Histograms for face recognization
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        #Get the Faces and IDs
        faces,ids = getImagesAndLabels('dataset', self)
        
        # Train the model using the faces and IDs
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer.yml
        recognizer.save('trainer/trainer.yml')
        self.label.setText("Learning Complete")

    def recognise(self):
        self.progressBar.setValue(0)
        fRecogniser(self)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Raspberry-Pi Face Recognition"))
        self.new_face.setText(_translate("MainWindow", "Add New Face"))
        self.train.setText(_translate("MainWindow", "Train Faces"))
        self.recog_face.setText(_translate("MainWindow", "Recognise Face"))
        self.label.setText(_translate("MainWindow", "Raspberry-Pi Face Recognition"))
        self.label_2.setText(_translate("MainWindow", "No. of Faces recorded:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    createDict()
    MainWindow.show()
    sys.exit(app.exec_())