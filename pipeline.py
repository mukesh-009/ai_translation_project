import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS

# 1. Load models
asr_model = whisper.load_model("tiny")
translator_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(translator_name)
translator = MarianMTModel.from_pretrained(translator_name)

# 2. Transcribe audio
result = asr_model.transcribe("your_audio.wav")
print("ASR:", result["text"])

# 3. Translate
inputs = tokenizer([result["text"]], return_tensors="pt", padding=True)
translated = translator.generate(**inputs)
output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print("Translated:", output_text)

# 4. Convert to speech
tts = gTTS(output_text, lang="hi")
tts.save("translated.mp3")
print("✅ Translation pipeline complete! Check translated.mp3")