import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename) -> None:
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        # load and prepare image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # predict the image
        results = np.argmax(model.predict(test_image), axis=1)

        if results[0] == 1:
            prediction = 'Healthy'
            return [{"image": prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{"image": prediction}]
        
