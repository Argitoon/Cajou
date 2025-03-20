# 1. Built-in Python libraries
import os
import sys
import shutil
import platform
import subprocess
from datetime import datetime

# 2. Third-party libraries
import cv2
import torch
import dropbox
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dropbox.files import WriteMode
import webbrowser
import yaml

## PyQt6 modules
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QImage, QFontDatabase, QIcon
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QSizePolicy, QStackedWidget, QLineEdit, QInputDialog
)

# 3. Project-specific imports
from classifier.tools import predict, load_model, validate_path, shorten_filename
from application.dropbox_tools import count_images_on_dropbox, import_images_from_dropbox

# Load the configuration file
with open("variables.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MODEL_TYPE = config["MODEL_TYPE"]
LOCAL_CAPTURE_PATH = config["LOCAL_PATH"]["CAPTURE"]
LOCAL_MODEL_PATH = config["LOCAL_PATH"]["APP_MODEL"]
LOCAL_DATA_PATH = config["LOCAL_PATH"]["APP_DATASET"]
REMOTE_MODEL_PATH = config["REMOTE_PATH"]["MODEL"]
REMOTE_DATA_PATH = config["REMOTE_PATH"]["DATASET"]

CLASS_NAMES = config["CLASS_NAMES"]
CLASS_NAMES_FR = config["CLASS_NAMES_FR"]
APP_KEY = config["APP_KEY"]
APP_SECRET = config["APP_SECRET"]
PID = config["PID_SAVE"]

# Image selection modes
WEBCAM_MODE = 0
FILE_MODE = 1

# List of possible flags for cv2.VideoCapture
BACKENDS = {
    "Windows": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW],  # DirectShow, Media Foundation, Video for Windows
    "Linux": [cv2.CAP_V4L2, cv2.CAP_ANY],  # Video for Linux 2, Fallback OpenCV
    "Darwin": [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]  # AVFoundation (macOS), Fallback OpenCV
}
system = platform.system()
backends_to_try = BACKENDS.get(system, [cv2.CAP_ANY])  # If system unknown, try any backend

# Constants
TXT_FONT = "Arial"
TXT_SIZE = 12
TITLE_SIZE = 24

LOGO_PATH = "application/images/logo.png"
MINILOGO_PATH = "application/images/minilogo.png"

WINDOW_SIZE = (800, 500)

class MainWindow(QWidget):
    def __init__(self, model):
        super().__init__()
        self.image_path = None
        self.cap = None
        self.save_path = ""
        self.select_image_mode = FILE_MODE
        self.class_buttons = []
        self.save_button = None
        self.class_index = None

        self.model = model
        self.setWindowTitle("Application de Classification de Jouets")
        
        self.setGeometry(100, 100, *WINDOW_SIZE)
        self.setWindowIcon(QIcon(MINILOGO_PATH))

        # Change the font of the application
        QFontDatabase.addApplicationFont("Roboto-Bold.ttf")
        QFontDatabase.addApplicationFont("Roboto-Light.ttf")

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: 'Roboto';
            }
            QLabel {
                color: #336633;
            }
            QPushButton {
                background-color: #ff6633;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc5522;
            }
            QPushButton:disabled {
                background-color: #ed9172;
                color: #dddddd;
            }
        """)
        
        self.pred_button_style = """
            QPushButton {
                background-color: #336633;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #285228;
            }
            QPushButton:disabled {
                background-color: #485e48;
                color: #d9d9d9;
            }
        """
        
        self.dropbox_page = self.create_dropbox_page()
        self.home_page = self.create_home_page()
        self.webcam_page = self.create_webcam_page()
        self.model_page = self.create_model_page()
        self.classification_page = self.create_classification_page()

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.dropbox_page)
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.classification_page)
        self.stack.addWidget(self.webcam_page)
        self.stack.addWidget(self.model_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

    def create_logo(self) :
        """
        Create the logo of the application.
        """
        logo = QLabel()
        logo.setPixmap(QPixmap(LOGO_PATH).scaled(400, 160, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return logo
        
    def create_button(self, text, handler, font = TXT_FONT, size = TXT_SIZE, weight = QFont.Weight.Normal):
        """
        Create a button with a given text and handler.
        """
        button = QPushButton(text)
        button.setFont(QFont(font, size, weight=weight))
        button.clicked.connect(handler)
        return button

    def create_text(self, text, font = TXT_FONT, size = TXT_SIZE, weight = QFont.Weight.Normal, alignment = Qt.AlignmentFlag.AlignCenter):
        """
        Create a label with a given text and font.
        """
        text_ = QLabel(text)
        text_.setFont(QFont(font, size, weight=weight))
        text_.setAlignment(alignment)
        return text_
    
    def create_placeholder(self, text, font = TXT_FONT, size = TXT_SIZE, weight = QFont.Weight.Normal):
        """
        Create a placeholder with a given text and font.
        """
        placeholder = QLineEdit()
        placeholder.setFont(QFont(font, size, weight=weight))
        placeholder.setPlaceholderText(text)
        return placeholder
    
    def dummy_function(self):
        """
        A dummy function that is used to initialize the handler of the next button (classification page).
        """
        pass

    def check_camera(self):
        """
        Check if a camera is available and return its index if found.
        """
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at the index {i}")
                cap.release()
                return i
            
        return None

    # 1. DROPBOX PAGE

    def create_dropbox_page(self):
        """
        Create the Dropbox page to connect the user to his Dropbox account in order to use 
        properly the application.
        """
        page = QWidget()
        layout = QVBoxLayout()
        
        logo = self.create_logo()
        layout.addWidget(logo)

        title = self.create_text("Classification Automatique de Jouets", size=TITLE_SIZE, weight=QFont.Weight.Bold)
        layout.addWidget(title)

        # Dropbox connection status
        self.dbx_connected = False
        self.dbx_status_text = self.create_text("Dropbox : Non connecté", size=TXT_SIZE)
        connect_dropbox_button = self.create_button("Se connecter à Dropbox", self.connect_to_dropbox, size=TXT_SIZE)

        layout.addWidget(connect_dropbox_button)
        layout.addWidget(self.dbx_status_text, alignment=Qt.AlignmentFlag.AlignCenter)
        
        page.setLayout(layout)
        return page

    # 2. HOME PAGE

    def create_home_page(self):
        """
        Create the home page with buttons to select an image, take a photo using the camera,
        and connect to Dropbox to upload images.
        """
        page = QWidget()
        layout = QVBoxLayout()
        
        logo = self.create_logo()
        layout.addWidget(logo)

        title = self.create_text("Classification Automatique de Jouets", size = TITLE_SIZE, weight=QFont.Weight.Bold)
        layout.addWidget(title)

        # Buttons
        self.select_button = self.create_button("Choisir une image sur cet appareil", self.select_image, size=TXT_SIZE)
        self.webcam_button = self.create_button("Prendre une photo", self.go_to_webcam_page, size=TXT_SIZE)
        
        if self.model == None:
            self.select_button.setEnabled(False)
            self.webcam_button.setEnabled(False)
        
        model_button = self.create_button("Entraînner CAJOU", self.go_to_model_page, size=TXT_SIZE)

        # Text
        self.save_path_text_homepage = self.create_text("", size=TXT_SIZE)

        layout.addWidget(self.select_button)
        layout.addWidget(self.webcam_button)
        layout.addWidget(model_button)
        layout.addWidget(self.save_path_text_homepage, alignment=Qt.AlignmentFlag.AlignCenter)
        
        page.setLayout(layout)
        return page

    def connect_to_dropbox(self):
        """
        Handles the Dropbox authentication flow.
        """

        auth_flow = DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)
        authorize_url = auth_flow.start()
        webbrowser.open(authorize_url)

        # Asking user's authentication
        code, ok = QInputDialog.getText(
            self,
            "Connexion Dropbox",
            "Collez le code d'autorisation Dropbox ici :"
        )

        if ok and code:
            try:
                oauth_result = auth_flow.finish(code)
                self.dbx = dropbox.Dropbox(oauth_result.access_token)
                self.dbx_connected = True
                self.dbx_status_text.setText("Dropbox : Connecté ✅")
                self.stack.setCurrentWidget(self.home_page)
            except Exception as e:
                self.dbx_connected = False
                self.dbx_status_text.setText(f"Erreur connexion Dropbox ❌ : {e}")

    def select_image(self):
        """
        Open a file dialog to select an image from the user's computer.
        """
        self.select_image_mode = FILE_MODE
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            self.image_path = file_path
            file_name = os.path.basename(file_path)
            file_name = shorten_filename(file_name, 30)
            self.save_path_text_homepage.setText(f"Fichier sélectionné : {file_name}")  
            self.image_name_input.setText(file_name)

            self.go_to_classification_page()

    def go_to_home_page(self):
        """
        Go back to the home page.
        """
        if self.model == None:
            self.select_button.setEnabled(False)
            self.webcam_button.setEnabled(False)
        else :
            self.select_button.setEnabled(True)
            self.webcam_button.setEnabled(True)
            
        self.reset_classif_page()
        self.stack.setCurrentWidget(self.home_page)

    # 3. WEBCAM PAGE
    def create_webcam_page(self):
        """
        Create the webcam page where the user can see in real-time the video, choose an image name and capture a snapshot.
        """
        page = QWidget()
        layout = QVBoxLayout()

        title = self.create_text("Capture de l'image en temps réel", size=TITLE_SIZE, weight=QFont.Weight.Bold)
        layout.addWidget(title)

        # To display the webcam
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Input field to rename image
        self.image_name_input = self.create_placeholder("Entrez le nom de l'image", size=TXT_SIZE)
        layout.addWidget(self.image_name_input, alignment=Qt.AlignmentFlag.AlignCenter)

        # Capture button to take a snapshot
        capture_snapshot_button = self.create_button("Capturer l'image", self.capture_snapshot, size=TXT_SIZE)
        layout.addWidget(capture_snapshot_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Menu button to go back to home page
        menu_button = self.create_button("Menu", self.go_to_home_page, size=TXT_SIZE)
        layout.addWidget(menu_button, alignment=Qt.AlignmentFlag.AlignCenter)

        page.setLayout(layout)

        self.timer = QTimer(self)  # Timer to refresh the video feed
        self.timer.timeout.connect(self.update_video_feed)

        return page

    def update_video_feed(self):
        """
        Update the video feed by reading a frame from the camera and displaying it in the video label.
        """
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to QImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Set the image to the pixmap
            pixmap = QPixmap.fromImage(q_img)

            # Get the current size of the video label (widget where the video is shown)
            label_width = self.video_label.width()
            label_height = self.video_label.height()

            # Calculate the scaling factor while maintaining the aspect ratio
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            # Calculate the scale factor to fit the label size
            scale_width = label_width / pixmap_width
            scale_height = label_height / pixmap_height
            scale_factor = max(scale_width, scale_height) * 0.5

            # Scale the pixmap to fit within the label
            scaled_pixmap = pixmap.scaled(int(pixmap_width * scale_factor), int(pixmap_height * scale_factor), Qt.AspectRatioMode.KeepAspectRatio)

            # Set the scaled pixmap to the video label
            self.video_label.setPixmap(scaled_pixmap)

    def capture_snapshot(self):
        """
        Capture the current frame from the webcam and save it to a file with a timestamp or the provided name.
        """
        ret, frame = self.cap.read()
        if ret:
            image_name = self.image_name_input.text().strip()

            # If no name provided, use timestamp
            if not image_name:
                image_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_path = LOCAL_CAPTURE_PATH + f"/{image_name}.jpg"
            validate_path(save_path)
            
            # Save the frame to the specified path
            self.image_path = save_path
            cv2.imwrite(save_path, frame)

            # Display the action
            self.save_path_text_homepage.setText(f"Image sauvegardée sous : {save_path}")
            print(f"Image sauvegardée sous : {save_path}")

            # Stop the timer to avoid a bug when reusing the webcam mode
            self.timer.stop()
            
            self.go_to_classification_page()

    def go_to_webcam_page(self):
        """
        Go to the webcam page and start the video capture.
        """
        self.select_image_mode = WEBCAM_MODE
        self.image_name_input.setText("")

        # Start the video capture
        camera_index = self.check_camera()
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print("Error: Unable to open the webcam")
            return
        
        # Start the timer to update the video feed
        self.timer.start(16)  # ~ 60 fps (not usefull to get more fps)
        self.stack.setCurrentWidget(self.webcam_page)
    
    # 4. CLASSIFICATION PAGE
    def create_classification_page(self):
        """
        Create the classification page with the image, classification text blocks, classification class and action buttons.
        """
        page = QWidget()
        layout = QVBoxLayout()

        title = self.create_text("Classification du jouet", size=TITLE_SIZE, weight=QFont.Weight.Bold)
        layout.addWidget(title)

        # I. Image block (left side of the screen)
        self.block1 = QLabel()
        self.block1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        box1 = QVBoxLayout()
        box1.addWidget(self.block1)

        # II. Classification text block (right side of the screen)
        self.block2 = QLabel()
        self.block2.setFont(QFont("Roboto", 15, QFont.Weight.Light))
        self.block2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        box2 = QVBoxLayout()
        box2.addWidget(self.block2)
        
        # The 2 blocks together (on the save height on the screen)
        boxes = QHBoxLayout()
        boxes.addLayout(box1)
        boxes.addLayout(box2)

        layout.addLayout(boxes)
        self.save_path_text_classif_page = self.create_text(self.save_path, size=TXT_SIZE)
        layout.addWidget(self.save_path_text_classif_page, alignment=Qt.AlignmentFlag.AlignCenter)

        # Class buttons
        class_buttons_layout = QHBoxLayout()
        
        for classn in CLASS_NAMES_FR:
            index = CLASS_NAMES_FR.index(classn)
            button = self.create_button(classn, lambda checked, i=index: self.choose_class(i), size=TXT_SIZE)
            self.class_buttons.append(button)
            class_buttons_layout.addWidget(button)

        
        # Add a button "Hors Catégorie" to allow the user to not classify the image
        button_hors_categorie = self.create_button("Hors Catégorie", lambda checked, i=len(CLASS_NAMES_FR): self.choose_class(i), size=TXT_SIZE)
        self.class_buttons.append(button_hors_categorie)
        class_buttons_layout.addWidget(button_hors_categorie)

        for button in self.class_buttons:
            button.setStyleSheet(self.pred_button_style)
        
        layout.addLayout(class_buttons_layout)

        # III. Menu, Cancel, Save and Next buttons (bottom of the screen)
        action_buttons_layout = QHBoxLayout()

        menu_button = self.create_button("Menu", self.go_to_home_page)
        self.cancel_button = self.create_button("Annuler", self.cancel)
        self.save_button = self.create_button("Sauvegarder", self.save)
        self.next_button = self.create_button("Suivant", self.dummy_function)

        # Disable the Cancel and Save buttons initially
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(False)

        action_buttons_layout.addWidget(menu_button, alignment=Qt.AlignmentFlag.AlignCenter)
        action_buttons_layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignCenter)
        action_buttons_layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignCenter)
        action_buttons_layout.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(action_buttons_layout)

        page.setLayout(layout)
        return page

    def choose_class(self, class_index):
        """
        Display in green the selected class and enables to cancel the action or save the image.
        """
        self.class_index = class_index
        self.class_buttons[class_index].setStyleSheet("background-color: #57d957; color: #262626;")

        for button in self.class_buttons:
            button.setEnabled(False)

        self.cancel_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def cancel(self):
        """
        Cancel the last action : the class selection or the image saved.
        """
        # If the image has been saved, then we remove it
        if self.save_path_text_classif_page.text().strip() != "":
            try:
                self.dbx.files_delete(self.save_path)
                print(f"File deleted : {self.save_path}")
            except:
                print(f"Error while deleting the file : {self.save_path}")

        self.reset_classif_page()

    def save_image(self):
        """
        Save the image to a folder corresponding to the class selected.
        """
        if self.class_index >= len(CLASS_NAMES):
            class_name = "Hors_Categorie"
            print(class_name)
            folder_name = os.path.join(REMOTE_DATA_PATH, "Hors_Categorie")
        else:
            class_name = CLASS_NAMES[self.class_index]
            folder_name = os.path.join(REMOTE_DATA_PATH, class_name)

        folder_name = folder_name.replace("\\", "/")

        try:
            # Create the folder on Dropbox if necessary
            self.dbx.files_create_folder_v2(folder_name)
        except dropbox.exceptions.ApiError as err:
            #print(f"Folder {folder_name} already exists.")
            pass
        except Exception as e:
            print(f"An unexpected error occurred while creating the folder : {e}")
            pass

        # Count the number of existing images in the folder
        image_count = count_images_on_dropbox(self.dbx, folder_name)

        # Create the new image name based on class name and the next available index
        image_name = f"{class_name}_{image_count + 1}"
        self.save_path = os.path.join(folder_name, f"{image_name}.jpg")
        
        self.save_path = self.save_path.replace("\\", "/")
        
        if os.path.exists(self.image_path):
            try:
                # Ouvrir l'image source
                with open(self.image_path, "rb") as f:
                    self.dbx.files_upload(f.read(), self.save_path, mode=WriteMode("overwrite"))
                
                # Update the texts with the save path
                txt = f"Image sauvegardée sous : {self.save_path}"
                self.save_path_text_homepage.setText(txt)
                self.save_path_text_classif_page.setText(txt)
                print(txt)
            except Exception as e:
                print(f"Error while saving the image : {e}")
        else:
            print(f"Source image {self.image_path} not found.")
            self.save_path_text_homepage.setText("Erreur : L'image source n'a pas été trouvée.")
            self.save_path_text_classif_page.setText("Erreur : L'image source n'a pas été trouvée.")
              
    def save(self):
        """
        Process the saving and reset the state of the save button.
        """
        self.save_button.setEnabled(False)
        self.save_image()
          
    def next_FILE_MODE(self):
        """
        Redirect to the select image page.
        """
        self.reset_classif_page()
        self.select_image()

    def next_WEBCAM_MODE(self):
        """
        Redirect to the webcam image capture page.
        """
        self.reset_classif_page()
        self.go_to_webcam_page()

    def update_next_button(self):
        """
        Update the next button in function of the mode selected by the user in the menu.
        """
        if self.select_image_mode == FILE_MODE:
            self.next_button.clicked.disconnect()  # Disconnect any previous function
            self.next_button.clicked.connect(self.next_FILE_MODE)  # Connect the function 'select_image'
        elif self.select_image_mode == WEBCAM_MODE:
            self.next_button.clicked.disconnect()  # Disconnect any previous function
            self.next_button.clicked.connect(self.next_WEBCAM_MODE)  # Connect the function 'capture_snapshot'

    def reset_classif_page(self):
        """
        Reset the states of the buttons and the text in the classification page and the save path of the menu.
        """
        # Buttons
        if self.class_buttons is not None and self.class_index is not None:
            self.class_buttons[self.class_index].setStyleSheet(self.pred_button_style)

            for button in self.class_buttons:
                button.setEnabled(True)

        if self.cancel_button is not None:
            self.cancel_button.setEnabled(False)

        if self.save_button is not None:
            self.save_button.setEnabled(False)

        # Texts
        if self.save_path_text_homepage is not None:
            self.save_path_text_homepage.setText("")

        if self.save_path_text_classif_page is not None:
            self.save_path_text_classif_page.setText("")

    def go_to_classification_page(self):
        """
        Load the image and set the classification results on the classification page.
        """
        top = 5

        if self.image_path:
            image = QPixmap(self.image_path) # Load the image

            window_size = self.size()
            new_width = window_size.width() // 3
            new_height = window_size.height() // 2

            # Resize the image (left side of the screen)
            if not image.isNull():
                self.block1.setPixmap(image.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.block1.setFixedSize(new_width, new_height)
            
            # Classification results
            classes, probabilities = predict(self.model, self.image_path, top=top)
            
            # Saving the results in a string
            prediction_text = ""
            for i in range(top):
                fr_class = CLASS_NAMES_FR[CLASS_NAMES.index(classes[i])]
                prediction_text += "{} : {:.2f}%\n".format(fr_class, probabilities[i] * 100)

            # Set the string on the right side of the screen
            self.block2.setText(prediction_text)
            self.stack.setCurrentWidget(self.classification_page)

        self.update_next_button()
        
    # 5. MODEL PAGE
    def create_model_page(self):
        """
        Create the model page with a button to start the training.
        """
        page = QWidget()
        layout = QVBoxLayout()

        title = self.create_text("Entraînement du Modèle", size=TITLE_SIZE, weight=QFont.Weight.Bold)
        layout.addWidget(title)
        
        # I. Number of images saved in Dropbox
        self.nb_images_dbx = None
        self.image_dbx_txt = self.create_text(f"Nombre d'images utilisables pour l'entraînement dans Dropbox : {self.nb_images_dbx}", size=TXT_SIZE)
        layout.addWidget(self.image_dbx_txt, alignment=Qt.AlignmentFlag.AlignCenter)

        # II. Model training buttons
        model_training_layout = QVBoxLayout()
        
        self.model_txt = self.create_text("Entraîner CAJOU pour améliorer sa précision.")
        self.model_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_training_layout.addWidget(self.model_txt)
        
        self.train_button = self.create_button("Lancer l'entraînement", self.start_train)
        self.stop_button = self.create_button("Arrêter l'entraînement", self.stop_train)
        self.upload_button = self.create_button("Upload le modèle", self.upload_model)
        self.download_button = self.create_button("Télécharger le modèle", self.download_model)
        
        model_training_layout.addWidget(self.train_button, alignment=Qt.AlignmentFlag.AlignCenter)
        model_training_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
        model_training_layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignCenter)
        model_training_layout.addWidget(self.download_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(model_training_layout)
        
        # III. Return button
        return_button = self.create_button("Retour", self.go_to_home_page, size=TXT_SIZE)
        layout.addWidget(return_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        page.setLayout(layout)
        return page
    
    def update_nb_images_dbx(self):
        nb_images = 0
        for class_name in CLASS_NAMES:
            folder_name = os.path.join(REMOTE_DATA_PATH, class_name).replace("\\", "/")
            try:
                nb_images += count_images_on_dropbox(self.dbx, folder_name)
            except dropbox.exceptions.ApiError as _:
                print(f"Path {folder_name} not found in Dropbox")
                continue
        
        self.nb_images_dbx = nb_images
        self.image_dbx_txt.setText(f"Nombre d'images utilisables pour l'entraînement dans Dropbox : {self.nb_images_dbx}")
    
    def prepare_training(self):
        """
        Copy the saved images and the training dataset to a new training folder and start the training.
        """
        print("Transfert has started...")
        
        remote_folder = REMOTE_DATA_PATH
        local_folder = LOCAL_DATA_PATH
        
        # Reinitalize the local folder
        if os.path.exists(local_folder):
            shutil.rmtree(local_folder)
        
        # Move the saved images to the training dataset folder
        for class_name in CLASS_NAMES :
            remote_class_folder = remote_folder + "/" + class_name
            local_class_folder = local_folder + "/" + class_name
            
            validate_path(local_class_folder)
            import_images_from_dropbox(self.dbx, remote_class_folder, local_class_folder)
      
    def start_train(self) :
        """
        Start the training of the model in a separate process.
        """
        print("Starting training...")
        
        if not self.is_model_training():
            self.prepare_training()
            self.training_process = subprocess.Popen([sys.executable, "main.py", "apptrain"])
            
            # Save the PID of the training process
            with open(PID, "w") as f:
                f.write(str(self.training_process.pid))

            print(f"Training started (PID: {self.training_process.pid})")
            
            # Update the buttons
            self.train_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.model_txt.setText("Entraînement en cours...")
        
    def stop_train(self) :
        """
        Stop the training of the model.
        """
        pid = self.get_training_pid()
        if pid:
            try:
                os.kill(pid, 9)  # 9 = SIGKILL
                os.remove(PID)  # Remove the PID file
                print(f"Training stopped (PID: {pid})")
                
                # Update the buttons
                self.train_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.model_txt.setText("Entraînement arrêté.")
                
            except ProcessLookupError:
                print(f"Process {pid} not found.")
  
    def upload_model(self) :
        """
        Upload the model to Dropbox.
        """
        if is_model_trained():
            # Upload the model
            with open(LOCAL_MODEL_PATH, "rb") as f:
                self.dbx.files_upload(f.read(), REMOTE_MODEL_PATH, mode=WriteMode("overwrite"))
            print("Model uploaded.")
            
            # Update the buttons
            self.upload_button.setEnabled(False)
            self.download_button.setEnabled(True)
            self.model_txt.setText("Modèle uploadé.")
        else:
            print("Error : Upload button should not be accessible.")
    
    def download_model(self) :
        """
        Download the model from Dropbox.
        """
        
        if self.is_model_online():
            # Download the model
            validate_path(LOCAL_MODEL_PATH)
            with open(LOCAL_MODEL_PATH, "wb") as f:
                _, res = self.dbx.files_download(REMOTE_MODEL_PATH)
                f.write(res.content)
            print("Model downloaded.")
            
            # Reload the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            self.model = load_model(MODEL_TYPE, LOCAL_MODEL_PATH).to(device)
            
            # Update the buttons
            self.download_button.setEnabled(False)
            self.upload_button.setEnabled(True)
            self.model_txt.setText("Modèle téléchargé.")
        else:
            print("Error : Download button should not be accessible.")
    
    def get_training_pid(self):
        """
        Récupère le PID du fichier.
        """
        if os.path.exists(PID):
            with open(PID, "r") as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return None
        return None
    
    def is_model_training(self) -> bool :
        """
        Check if the model is currently training.
        """

        # Get the PID of the training process
        pid = self.get_training_pid()
        
        # Check if the process is still running
        if pid is not None:
            if platform.system() == "Windows":
                result = subprocess.run(["tasklist", "/fi", f"PID eq {pid}"], capture_output=True)
                if "python.exe" in result.stdout.decode(encoding="utf-8", errors="ignore"):
                    return True
            else:
                result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
                if "python" in result.stdout.decode(encoding="utf-8", errors="ignore"):
                    return True
        return False

    def is_model_online(self) -> bool:
        """
        Check if the model is online.
        """
        # Check in dropbox if the model exists
        try:
            self.dbx.files_get_metadata(REMOTE_MODEL_PATH)
            return True
        except dropbox.exceptions.ApiError as _:  # File not found
            return False
    
    def go_to_model_page(self):
        """
        Go to the model page.
        """
        self.stack.setCurrentWidget(self.model_page)
        
        # Update the number of images saved in Dropbox
        self.update_nb_images_dbx()
        
        # Enable/Disable training button
        if self.nb_images_dbx == None or self.nb_images_dbx < 1 or self.is_model_training() :
            self.train_button.setEnabled(False)
        else :
            self.train_button.setEnabled(True)
        
        # Enable/Disable stop button
        if self.is_model_training() :
            self.stop_button.setEnabled(True)
            self.model_txt.setText("Entraînement en cours...")
        else :
            self.stop_button.setEnabled(False)
        
        # Enable/Disable upload button
        if is_model_trained() :
            self.upload_button.setEnabled(True)
        else :
            self.upload_button.setEnabled(False)
            
        # Enable/Disable download button
        if self.is_model_online() :
            self.download_button.setEnabled(True)
        else :
            self.download_button.setEnabled(False)
        
def is_model_trained() -> bool:
        """
        Check if there is a trained model in the specified path.
        """
        # Check if the model file exists
        if os.path.exists(LOCAL_MODEL_PATH):
            return True
        return False
    
def main() -> None :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try :
        model = load_model(model_type = MODEL_TYPE, model_path = LOCAL_MODEL_PATH).to(device)
    except :
        model = None
        print(f"There is no model to load in {LOCAL_MODEL_PATH}.")     
     
    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py application\'")