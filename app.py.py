import streamlit as st
import sounddevice as sd
import soundfile as sf
import os
import tempfile
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import requests  # For HTTP communication with Raspberry Pi
import asyncio
import edge_tts

# Load environment variables
find_dotenv()
load_dotenv()

# Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Functions for fan control ---
# Removed GPIO, replaced with HTTP request to control fan on Raspberry Pi

def send_fan_command(state):
    try:
        # Assuming the Raspberry Pi is running a Flask server on port 5000
        response = requests.post("http:192.168.234.72:5000/fan", json={"state": state})
        if response.status_code == 200:
            return f"Fan turned {state.upper()} successfully!"
        else:
            return f"Failed to turn {state} fan. Error: {response.status_code}"
    except Exception as e:
        return f"Error communicating with Raspberry Pi: {e}"

# Functions for voice assistant
def record_audio(duration=5, sample_rate=44100):
    st.write("🎙️ Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.write("✅ Recording complete")
    return audio_data, sample_rate

def save_audio(audio_data, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sample_rate)
    return temp_file.name

def transcribe_audio_with_whisper(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en"
            )
            return response.text
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

def get_groq_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a helpful voice assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        return None

def text_to_speech(text):
    try:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        voice = "en-US-GuyNeural"  # Male voice
        asyncio.run(edge_tts.Communicate(text, voice).save(output_path))
        return output_path
    except Exception as e:
        st.error(f"❌ Text-to-speech error: {e}")
        return None

# --- UI Setup ---
st.set_page_config(page_title="Vision Assistant", layout="centered")
st.title("🔮 Digital Voice Assistant - Vision")
st.markdown("---")

# Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("🗣️ Your Input")
with col2:
    st.subheader("🤖 Assistant Output")

# --- Audio Input Option ---
if st.button("🔊 Speak Now (5 sec)"):
    audio_data, sample_rate = record_audio()
    audio_file = save_audio(audio_data, sample_rate)

    with col1:
        st.audio(audio_file, format="audio/wav")

    transcription = transcribe_audio_with_whisper(audio_file)
    os.unlink(audio_file)

    if transcription:
        with col1:
            st.markdown(f"🧙💬 **You:** {transcription}")

        # Check for fan control commands
        if any(keyword in transcription.lower() for keyword in ["turn on fan", "start fan", "activate fan"]):
            response = send_fan_command("on")  # Send HTTP command to Raspberry Pi
        elif any(keyword in transcription.lower() for keyword in ["turn off fan", "stop fan", "deactivate fan"]):
            response = send_fan_command("off")  # Send HTTP command to Raspberry Pi
        else:
            # Forward non-fan commands to Groq API (i.e., general queries or other tasks)
            response = get_groq_response(transcription)

        if response:
            with col2:
                st.markdown(f"🤖 **Assistant:** {response}")
                speech_file = text_to_speech(response)
                if speech_file:
                    st.audio(speech_file, format="audio/mp3")
                    os.unlink(speech_file)
