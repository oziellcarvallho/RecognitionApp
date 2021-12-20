__version__ = "1.0"

from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.logger import Logger
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from cv2 import cvtColor, COLOR_RGBA2GRAY
from kivy.graphics.context_instructions import PopMatrix, PushMatrix, Rotate
from image_processing.recognition import classify
import numpy as np
from plyer import tts

class Recognition(App):
    def build(self):
        self.floatLayout = FloatLayout()
        self.floatLayout.orientation = 'vertical'
        
        self.camera = Camera()
        self.camera.resolution = (640, 480)
        self.camera.size = (Window.height, Window.width)
        self.camera.allow_stretch = True
        self.camera.keep_ratio = True
        self.camera.play = True
        
        with self.camera.canvas.before:
            PushMatrix()
            self.rot = Rotate()
            self.rot.angle = -90
            self.rot.origin = (self.camera.center_y, self.camera.center_x)
        with self.camera.canvas.after:
            PopMatrix()
        
        self.button = Button()
        self.button.text = 'Capture'
        self.button.size_hint = (1, .1)
        self.button.bind(on_press = self.capture)

        self.floatLayout.add_widget(self.camera)
        self.floatLayout.add_widget(self.button)

        return self.floatLayout

    def get_image(self):
        height, width = self.camera.texture.height, self.camera.texture.width
        image_buffer = np.frombuffer(self.camera.texture.pixels, np.uint8)
        image_buffer = image_buffer.reshape(height, width, 4)
        gray = cvtColor(image_buffer, COLOR_RGBA2GRAY)
        return gray

    def capture(self, args):
        try:
            image = self.get_image()
            result = classify(image)
            tts.speak(result)
            Logger.info('Recognition: '+ result)
        except Exception:
            tts.speak('Erro na classificação')
            Logger.exception('Recognition: >>> Erro <<<')

if __name__ == '__main__':
    Recognition().run()