"""เว็บจำแนกสัตว์หิมพานต์ ใช้ดีไซน์เดียวกับหน้า Apps Script"""

import base64
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
ROOT = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(ROOT, "himmapan_cnn.weights.h5")
himmapan_ui = components.declare_component(
    "himmapan_ui",
    path=os.path.join(ROOT, "frontend"),
)


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


def predict_data_url(data_url):
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    resized = image.resize(IMG_SIZE)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probabilities = load_model().predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[best],
        "probs": {
            name: float(value)
            for name, value in zip(CLASS_NAMES, probabilities)
        },
    }


st.set_page_config(
    page_title="จำแนกสัตว์หิมพานต์",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      #MainMenu, [data-testid="stHeader"], [data-testid="stToolbar"],
      [data-testid="stDecoration"] { display: none !important; }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #f7f1e6;
      }
      .block-container, [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        max-width: 100% !important;
      }
      iframe { width: 100% !important; border: 0; }
      footer { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "seen_nonce" not in st.session_state:
    st.session_state.seen_nonce = None

weights_ready = os.path.isfile(WEIGHTS_PATH)
if not weights_ready and st.session_state.result is None:
    st.session_state.result = {
        "error": "ยังไม่มีไฟล์ himmapan_cnn.weights.h5",
        "key": "missing-weights",
    }

health = (
    "โมเดลพร้อมแล้ว อัปโหลดหรือถ่ายภาพ แล้วกดทำนาย"
    if weights_ready
    else "ยังไม่มีไฟล์น้ำหนักโมเดลบนเซิร์ฟเวอร์"
)

event = himmapan_ui(
    result=st.session_state.result,
    health=health,
    default=None,
    key="himmapan",
)

if (
    weights_ready
    and isinstance(event, dict)
    and event.get("image")
    and event.get("nonce") != st.session_state.seen_nonce
):
    st.session_state.seen_nonce = event["nonce"]
    try:
        result = predict_data_url(event["image"])
        result["key"] = str(event["nonce"])
        st.session_state.result = result
    except Exception as exc:
        st.session_state.result = {
            "error": "ทำนายไม่สำเร็จ: " + str(exc),
            "key": str(event["nonce"]),
        }
    st.rerun()
