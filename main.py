from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.uix.image import Image

from keras.models import load_model
from PIL import Image as PILImage, ImageOps
import numpy as np

class ImageRecognitionApp(BoxLayout):
    def __init__(self, **kwargs):
        super(ImageRecognitionApp, self).__init__(**kwargs)
        self.orientation = 'vertical'

        self.file_chooser = FileChooserListView()
        self.upload_button = Button(text='Upload Image', on_press=self.upload_image)
        self.result_label = Label(text='Recognition Result: ')
        self.progress_bar = ProgressBar(max=100)

        self.add_widget(self.file_chooser)
        self.add_widget(self.upload_button)
        self.add_widget(self.result_label)
        self.add_widget(self.progress_bar)

        self.model = load_model("keras_Model.h5", compile=False)
        self.class_names = open("labels.txt", "r").readlines()
        self.image_data = None

    def upload_image(self, instance):
        if not self.file_chooser.selection:
            self.show_error_popup("Please select an image first.")
            return

        image_path = self.file_chooser.selection[0]
        self.load_image(image_path)
        self.run_recognition()

    def load_image(self, image_path):
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load and preprocess the image
        image = PILImage.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, PILImage.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        self.image_data = data

    def run_recognition(self):
        if self.image_data is None:
            return

        # Predicts the model
        prediction = self.model.predict(self.image_data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Update the UI with the result
        self.result_label.text = f'Recognition Result: {class_name[2:]} (Confidence Score: {confidence_score})'
        self.progress_bar.value = 100

    def show_error_popup(self, message):
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))
        content.add_widget(Button(text='OK', on_press=lambda x: popup.dismiss()))

        popup = Popup(title='Error', content=content, size_hint=(0.4, 0.4))
        popup.open()

class MyApp(App):
    def build(self):
        return ImageRecognitionApp()

if __name__ == '__main__':
    MyApp().run()
