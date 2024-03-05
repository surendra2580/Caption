import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import pickle,os
from gtts import gTTS
from io import BytesIO

# Load the trained model and tokenizer
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'best_model.h5')
tokenizer_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'tokenizer.pkl')   

model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to convert an integer to a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break   
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to extract features from an image
def extract_features(image, vgg_model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image)
    return feature


# Function to generate audio from text
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

if __name__ == '__main__':
    # Streamlit app
    st.title('Image Captioning with Streamlit')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger caption generation
        # Button to trigger caption generation and audio synthesis
    if st.button('Generate Caption and Audio'):
        st.write("Generating caption and audio...")
        # Extract features from the uploaded image
        feature = extract_features(image, vgg_model)
        # Generate caption
        caption = predict_caption(model, feature, tokenizer, max_length=35).rstrip('endseq').lstrip('startseq').strip()  # Use the correct max_length
        st.write("Caption:", caption)
        # Generate audio
        audio_buffer = text_to_audio(caption)
        st.audio(audio_buffer, format='audio/mp3')
