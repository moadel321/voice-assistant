import keyboard
import sounddevice as sd
import numpy as np
import queue
import requests
import io
import soundfile as sf
import json
import arabic_reshaper
from bidi.algorithm import get_display
import time
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import csv
from datetime import datetime
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Hugging Face inference endpoint URL
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_ENDPOINT = "https://mrholocg8pxhkacd.us-east-1.aws.endpoints.huggingface.cloud"

# Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ElevenLabs voice ID
ELEVENLABS_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1

# Initialize a queue for audio data
audio_queue = queue.Queue()

def measure_latency(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, latency
    return wrapper

def display_arabic(text):
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        print(f"Error reshaping text: {e}")
        return text  # Return original text if reshaping fails

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def record_audio():
    """Record audio while spacebar is pressed."""
    print("Press and hold the spacebar to speak...")
    audio_queue.queue.clear()  # Clear the queue before recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            if keyboard.is_pressed('space'):
                print("Recording... Release spacebar to stop.")
                while keyboard.is_pressed('space'):
                    sd.sleep(100)
                return np.concatenate(list(audio_queue.queue))

@measure_latency
def transcribe_audio(audio):
    """Transcribe audio using Hugging Face dedicated inference endpoint."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "audio/flac"
    }
    
    # Ensure audio is a numpy array
    if not isinstance(audio, np.ndarray):
        audio = np.frombuffer(audio, dtype=np.float32)
    
    # Normalize audio to float32 range [-1, 1]
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    # Convert audio to FLAC format
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='flac')
    flac_data = buffer.getvalue()
    
    try:
        response = requests.post(HUGGINGFACE_ENDPOINT, headers=headers, data=flac_data)
        response.raise_for_status()
        response_json = response.json()
        transcription = response_json.get("text", None)
        if transcription is None:
            raise ValueError("Transcription key 'text' not found in response.")
        return transcription
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error: {e}")
        if response.text:
            print(f"Response content: {response.text}")
        return None

@measure_latency
def get_llm_response(transcription):
    """Get response from Groq's llama3-8b-8192 model."""
    if transcription is None:
        return "I'm sorry, I couldn't transcribe the audio. Could you please try again?"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. who speaks only in Egyptian arabic and no other language"},
            {"role": "user", "content": transcription}
        ]
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()['choices'][0]['message']['content']

@measure_latency
def text_to_speech(text: str) -> IO[bytes]:
    """Convert text to speech using ElevenLabs' API."""
    try:
        # Perform the text-to-speech conversion
        response = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # Create a BytesIO object to hold the audio data in memory
        audio_stream = BytesIO()

        # Write each chunk of audio data to the stream
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        # Reset stream position to the beginning
        audio_stream.seek(0)

        # Play the audio
        audio_segment = AudioSegment.from_mp3(audio_stream)
        audio_array = np.array(audio_segment.get_array_of_samples())
        sd.play(audio_array, samplerate=audio_segment.frame_rate)
        sd.wait()  # Wait until the audio is finished playing

        return audio_stream
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None

def write_to_csv(data, filename="latency_log.csv"):
    fieldnames = ["timestamp", "turn", "transcription_latency", "llm_latency", "tts_latency", "total_latency", "highest_latency"]
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def main():
    turn = 0
    while True:
        turn += 1
        # Record audio
        audio_data = record_audio()
        print("Audio recorded. Transcribing...")

        # Measure overall latency
        overall_start_time = time.time()

        # Transcribe audio
        transcription, transcription_latency = transcribe_audio(audio_data)
        print(f"Transcription: {display_arabic(transcription)}")

        # Get LLM response
        llm_response, llm_latency = get_llm_response(transcription)
        print(f"LLM Response: {display_arabic(llm_response)}")

        # Convert LLM response to speech
        print("Converting response to speech...")
        tts_result, tts_latency = text_to_speech(llm_response)

        # Calculate overall latency
        overall_latency = (time.time() - overall_start_time) * 1000  # Convert to milliseconds

        # Determine which API call took the most latency
        latencies = {
            "Transcription": transcription_latency,
            "LLM": llm_latency,
            "Text-to-Speech": tts_latency
        }
        highest_latency = max(latencies, key=latencies.get)

        # Prepare data for CSV
        data = {
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "transcription_latency": transcription_latency,
            "llm_latency": llm_latency,
            "tts_latency": tts_latency,
            "total_latency": overall_latency,
            "highest_latency": highest_latency
        }

        # Write to CSV
        write_to_csv(data)

        print(f"Overall latency: {overall_latency:.2f} ms")
        print(f"Highest latency: {highest_latency} ({latencies[highest_latency]:.2f} ms)")

        print("\nPress spacebar to speak again, or 'q' to quit.")
        if keyboard.read_key() == 'q':
            break

if __name__ == "__main__":
    main()