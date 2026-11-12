"""เว็บทำนายสัตว์หิมพานต์ สำหรับ Streamlit Community Cloud (ฟรี)

วางไฟล์ himmapan_cnn.weights.h5 ไว้โฟลเดอร์เดียวกับไฟล์นี้
ไฟล์นั้นอยู่ใน Google Drive:
งาน final_project_image_261_352/himmapan_cnn.weights.h5
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "พญานาค",
    "นกหัสดีลิงค์",
    "ครุฑ",
    "กินรี",
    "คชสีห์",
    "ช้างเอราวัณ",
    "สิงห์",
]
IMG_SIZE = (224, 224)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "himmapan_cnn.weights.h5")


def make_backbone():
    kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    try:
        return tf.keras.applications.EfficientNetB0(include_preprocessing=True, **kwargs)
    except TypeError:
        return tf.keras.applications.EfficientNetB0(**kwargs)


class HimmapanCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.backbone = make_backbone()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.head = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, images, training=False):
        features = self.backbone(tf.cast(images, tf.float32), training=False)
        return self.head(self.dropout(features, training=False))


@st.cache_resource
def load_model():
    model = HimmapanCNN(len(CLASS_NAMES), name="Himmapan_CNN")
    model.build((None, 224, 224, 3))
    model.load_weights(WEIGHTS_PATH)
    return model


def predict(image):
    resized = image.convert("RGB").resize(IMG_SIZE)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probabilities = load_model().predict(batch, verbose=0)[0]
    return probabilities


st.set_page_config(page_title="จำสัตว์หิมพานต์", page_icon="🐲")
st.title("จำสัตว์หิมพานต์")
st.caption("พญานาค · นกหัสดีลิงค์ · ครุฑ · กินรี · คชสีห์ · ช้างเอราวัณ · สิงห์")

if not os.path.isfile(WEIGHTS_PATH):
    st.error(
        "ยังไม่มีไฟล์ himmapan_cnn.weights.h5 ในโฟลเดอร์นี้ "
        "ดาวน์โหลดจาก Google Drive โฟลเดอร์ งาน final_project_image_261_352 แล้ววางคู่กับ app.py"
    )
    st.stop()

uploaded = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is None:
    st.info("เลือกรูปสัตว์หิมพานต์ แล้วเว็บจะบอกว่าเป็นตัวไหน")
    st.stop()

image = Image.open(uploaded)
st.image(image, caption=uploaded.name, width=360)

with st.spinner("กำลังทำนาย"):
    probabilities = predict(image)

best = int(np.argmax(probabilities))
st.subheader(f"ผลทำนาย: {CLASS_NAMES[best]}")
st.write(f"ความมั่นใจ {probabilities[best] * 100:.2f}%")

for index in np.argsort(probabilities)[::-1]:
    name = CLASS_NAMES[int(index)]
    percent = float(probabilities[index]) * 100
    st.write(f"{name} {percent:.2f}%")
    st.progress(min(1.0, percent / 100))
