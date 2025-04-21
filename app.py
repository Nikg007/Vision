import streamlit as st
import sounddevice as sd
import soundfile as sf
import os
import tempfile
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import requests
import asyncio
import edge_tts

# Load .env file
find_dotenv()
load_dotenv()

# Groq API
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Set this to your Raspberry Pi's IP address
RASPBERRY_PI_IP = "http://192.168.X.X:5000/fan"  # 🔁 Replace with your Pi's IP

# --- Fan control ---
def send_fan_command(state):
    try:
        response = requests.post(RASPBERRY_PI_IP, json={"state": state})
        if response.status_code == 200:
            return f"🌀 Fan turned {state.upper()}!"
        else:
            return f"⚠️ Error: {response.status_code}"
    except Exception as e:
        return f"❌ Cannot reach Raspberry Pi: {e}"

# --- Audio functions ---
def record_audio(duration=5, sample_rate=44100):
    st.write("🎙️ Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.write("✅ Recording complete")
    return audio, sample_rate

def save_audio(audio, sample_rate):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp.name, audio, sample_rate)
    return temp.name

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as f:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="en"
            )
            return response.text
    except Exception as e:
        st.error(f"🛑 Transcription failed: {e}")
        return None

def get_llm_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3-70b-8192",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"🛑 LLM error: {e}")
        return None

def text_to_speech(text):
    try:
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        voice = "en-US-GuyNeural"
        asyncio.run(edge_tts.Communicate(text, voice).save(path))
        return path
    except Exception as e:
        st.error(f"🛑 TTS failed: {e}")
        return None

# --- UI ---
st.set_page_config(page_title="Vision Assistant", layout="centered")
st.title("🧠 Vision Assistant")
st.markdown("---")

col1, col2 = st.columns(2)

if st.button("🎤 Speak Now (5 sec)"):
    audio, rate = record_audio()
    audio_file = save_audio(audio, rate)

    with col1:
        st.audio(audio_file, format="audio/wav")

    transcription = transcribe_audio(audio_file)
    os.unlink(audio_file)

    if transcription:
        with col1:
            st.markdown(f"💬 **You:** {transcription}")

        # Fan command check
        lower = transcription.lower()
        if any(kw in lower for kw in ["turn on fan", "start fan"]):
            result = send_fan_command("on")
        elif any(kw in lower for kw in ["turn off fan", "stop fan"]):
            result = send_fan_command("off")
        else:
            result = get_llm_response(transcription)

        if result:
            with col2:
                st.markdown(f"🤖 **Assistant:** {result}")
                mp3 = text_to_speech(result)
                if mp3:
                    st.audio(mp3, format="audio/mp3")
                    os.unlink(mp3)
