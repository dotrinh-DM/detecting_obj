import tensorflow as tf
import numpy as np
from utils import preprocess_image

# Load trained model
model = tf.keras.models.load_model('cnn_cifar10_model.h5')

# Label names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Predict
def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    pred_class = np.argmax(predictions)
    print(f"Đây là con: {class_names[pred_class]}")

# Ví dụ:
if __name__ == "__main__":
    predict('../img/img_1.png')
