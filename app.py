import streamlit as st
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os
import uuid
import traceback

st.set_page_config(page_title="AI Language Translator", layout="centered")
st.title("AI Language Translator (English ↔ Hindi)")

# Load models once and cache them so they are NOT reloaded on every interaction
@st.cache_resource
def load_asr_model(name="tiny"):
    return whisper.load_model(name)

@st.cache_resource
def load_translation_model(model_name="Helsinki-NLP/opus-mt-en-hi"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load models (cached)
asr_model = load_asr_model("tiny")
tokenizer, translation_model = load_translation_model()

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    tmp_wav = f"input_{uuid.uuid4().hex}.wav"
    out_mp3 = None
    try:
        # Save uploaded file to disk (Streamlit stores uploads in memory)
        with open(tmp_wav, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Transcribing audio — this can take a few seconds...")
        result = asr_model.transcribe(tmp_wav)
        transcript = result.get("text", "")
        st.write("ASR:", transcript)

        # Translate
        inputs = tokenizer([transcript], return_tensors="pt", padding=True)
        translated = translation_model.generate(**inputs)
        output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.write("Translation:", output_text)

        # TTS
        tts = gTTS(output_text, lang="hi")
        out_mp3 = f"out_{uuid.uuid4().hex}.mp3"
        tts.save(out_mp3)
        st.audio(out_mp3, format="audio/mp3")

    except Exception:
        st.error("An error occurred during processing. See details below:")
        st.text(traceback.format_exc())
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass
        try:
            if out_mp3 and os.path.exists(out_mp3):
                os.remove(out_mp3)
        except Exception:
            pass