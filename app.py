import streamlit as st
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS

st.title("AI Language Translator (English ↔ Hindi)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Save file
    with open("input.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # ASR
    asr_model = whisper.load_model("tiny")
    result = asr_model.transcribe("input.wav")
    st.write("ASR:", result["text"])

    # Translation
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer([result["text"]], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    st.write("Translation:", output_text)

    # TTS
    tts = gTTS(output_text, lang="hi")
    tts.save("out.mp3")
    st.audio("out.mp3")