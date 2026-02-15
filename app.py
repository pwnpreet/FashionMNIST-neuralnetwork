import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
st.set_page_config(page_title='Neural Networks',page_icon='🧠', layout='wide')

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://i.imgur.com/b9YGKmQ.jpeg");
    background-size: cover;
    background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

@st.cache_resource
def load_ann_model():
    return load_model("fashion_mnist_ann.keras")
model= load_ann_model()

class_names= ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f2f2f2;
}
.section {
    padding: 35px;
    margin: 40px auto;
    border-radius: 12px;
    max-width: 1200px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.09);
}
.section h2 {
    color: #1a1a1a;
    font-size: 34px;
    margin-bottom: 10px;
}
.section p {
    font-size: 17px;
    line-height: 1.7;
    color: #333333;
}
.tab-content {
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS ----------
def section(title, text):
    st.markdown(f"""
    <div class="section">
        <h2>{title}</h2>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

def tab_block(title, desc, uses):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {title}")
        st.markdown(desc)
    with col2:
        st.markdown("**Common Use Cases:**")
        st.markdown(uses)


# WELCOME SECTION-------------------
section(
    "🧠 Welcome to the Neural Network World",
    "Neural Networks are at the core of modern Artificial Intelligence, enabling machines "
    "to learn from data and make intelligent decisions inspired by the human brain."
)

c1, c2 = st.columns([1.2, 1])
with c1:
    section(
        "📌 What is this application about?",
        "This application explains Neural Networks conceptually and demonstrates how a "
        "trained model works in practice."
    )
with c2:
    section(
        "🚀 What will you explore?",
        "✔ Types of Neural Networks<br>"
        "✔ ANN Architecture<br>"
        "✔ Key components & terminologies<br>"
        "✔ Real-time model predictions"
    )

# TYPES OF NEURAL NETWORK-----------
section(
    "🧬 Types of Neural Networks",
    "Different neural network architectures are designed for different types of data "
    "and problem domains."
)

tabs = st.tabs(["ANN", "CNN", "RNN", "GANs", "Transformer"])
with tabs[0]:
    tab_block(
        "Artificial Neural Network (ANN)",
        "Fully connected network used mainly for structured and numerical data.",
        "- Classification\n- Regression\n- Risk prediction"
    )
with tabs[1]:
    tab_block(
        "Convolutional Neural Network (CNN)",
        "Designed for image and visual data using convolution operations.",
        "- Image classification\n- Face recognition\n- Medical imaging"
    )
with tabs[2]:
    tab_block(
        "Recurrent Neural Network (RNN)",
        "Handles sequential data by retaining past information.",
        "- Time-series forecasting\n- Text generation\n- Speech processing"
    )
with tabs[3]:
    tab_block(
        "Generative Adversarial Networks (GANs)",
        "Uses Generator and Discriminator networks to generate realistic data.",
        "- Image generation\n- Data augmentation\n- Art & deepfakes"
    )
with tabs[4]:
    tab_block(
        "Transformer Networks",
        "Uses attention mechanisms and parallel processing.",
        "- Chatbots\n- LLMs\n- Text summarization"
    )

# ANN ARCHITECTURE -----------------
section(
    "🧩 Artificial Neural Network (ANN) Architecture",
    "An ANN consists of interconnected layers that transform input data into output "
    "through weighted connections and activation functions."
)

with st.expander("🔍 View ANN Architecture Diagram & Explanation", expanded=False):
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.image(
            "img.png",  # <-- replace with your image path
            caption="Basic Architecture of an Artificial Neural Network",
            use_container_width=True
        )
    with col2:
        st.markdown("""
        **1️⃣ Input Layer**  
        This layer receives raw input features from the dataset. Each neuron represents one feature, and no computation is performed at this stage.
        **2️⃣ Hidden Layers**  
        Hidden layers perform the core computation. Each neuron applies weights,adds bias, and passes the result through an activation function to learn complex patterns in the data.
        **3️⃣ Output Layer**  
        The output layer produces the final prediction. The number of neurons depends on the problem type, such as binary classification, multi-class classification, or regression.
        """)

# KEY COMPONENTS---------
st.markdown("""
<div class="section">
    <h2>🧠 Key Components & Terminologies</h2>
    <p>
        Neural Networks are built using several important components that control how data flows through the model and how learning takes place. Understanding these terms is essential for designing and training effective neural networks.
    </p>
    <h3>🔹 Activation Functions</h3>
    <p>
        Activation functions introduce non-linearity into neural networks, enabling them to learn complex relationships beyond linear patterns.
    </p>
    <ul>
        <li><b>Sigmoid:</b> Compresses output between 0 and 1, commonly used in binary classification.</li>
        <li><b>ReLU:</b> Outputs zero for negative values and the input itself for positive values, improving training speed.</li>
        <li><b>Softmax:</b> Converts outputs into probability distributions, mainly used in multi-class classification.</li>
    </ul>
    <h3>🔹 Pooling</h3>
    <p>
        Pooling layers reduce the spatial dimensions of feature maps, helping to decrease computation and control overfitting. Max pooling is the most commonly used pooling technique.
    </p>
    <h3>🔹 Padding</h3>
    <p>
        Padding adds extra pixels (usually zeros) around input data to preserve spatial dimensions and prevent loss of information at the edges during convolution.
    </p>
    <h3>🔹 Optimization</h3>
    <p>
        Optimization algorithms adjust model weights to minimize loss during training. Popular optimizers include Gradient Descent, Adam, and RMSprop.
    </p>
    <h3>🔹 Flatten Layer</h3>
    <p>
        The flatten layer converts multi-dimensional data into a one-dimensional vector, making it suitable for fully connected layers in a neural network.
    </p>
    <h3>🔹 Dropout Layer</h3>
    <p>
        Dropout randomly disables a fraction of neurons during training to reduce overfitting and improve model generalization.
    </p>
    <h3>🔹 Batch Size</h3>
    <p>
        Batch size defines the number of training samples processed before the model updates its weights. Choosing the right batch size impacts training speed and model performance.
    </p>
</div>
""", unsafe_allow_html=True)

# MODEL PREDICTION SECTION ---------------
st.markdown("""
<div class="section">
    <h2>🔍 Model Prediction</h2>
    <p>
        Upload a Fashion MNIST image to see how the trained neural network predicts the class based on learned patterns.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload a Fashion MNIST image (28x28 grayscale)",
    type=['png', 'jpg', 'jpeg']
)
if uploaded_file is not None:
    st.markdown("<h4>📷 Uploaded Image</h4>", unsafe_allow_html=True)
    st.image(uploaded_file, width=200)

    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.markdown(f"""
    <div class="section">
        <h3>✅ Prediction Result</h3>
        <p style="font-size:18px;">
            <b>Predicted Class:</b> {class_names[predicted_class]}<br>
            <b>Confidence:</b> {confidence:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

