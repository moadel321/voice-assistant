import keyboard
import sounddevice as sd
import numpy as np
from openai import OpenAI
import queue
import threading
import requests
import io
import soundfile as sf
import json
import arabic_reshaper
from bidi.algorithm import get_display
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Hugging Face inference endpoint URL
HUGGINGFACE_ENDPOINT = "https://ejeic7mu20f8nbac.us-east-1.aws.endpoint.cloud"

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1

# Add this class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Initialize a queue for audio data
audio_queue = queue.Queue()

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
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            if keyboard.is_pressed('space'):
                print("Recording... Release spacebar to stop.")
                while keyboard.is_pressed('space'):
                    sd.sleep(100)
                return np.concatenate(list(audio_queue.queue))

import json

def transcribe_audio(audio):
    """Transcribe audio using Hugging Face dedicated inference endpoint."""
    start_time = time.time()
    
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
        transcription = response.json()["text"]
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Transcription latency: {latency:.2f} ms")
        return transcription
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if response.text:
            print(f"Response content: {response.text}")
        return None

def get_gpt4_response(transcription):
    """Get response from GPT-4 using OpenAI's API."""
    start_time = time.time()
    
    if transcription is None:
        return "I'm sorry, I couldn't transcribe the audio. Could you please try again?"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcription}
        ]
    )
    gpt4_response = response.choices[0].message.content
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"GPT-4 response latency: {latency:.2f} ms")
    
    return gpt4_response

def main():
    while True:
        # Record audio
        audio_data = record_audio()
        print("Audio recorded. Transcribing...")

        # Measure overall latency
        overall_start_time = time.time()

        # Transcribe audio
        transcription = transcribe_audio(audio_data)
        print(f"Transcription: {display_arabic(transcription)}")

        # Get GPT-4 response
        gpt4_response = get_gpt4_response(transcription)
        print(f"GPT-4 Response: {display_arabic(gpt4_response)}")

        # Calculate and print overall latency
        overall_end_time = time.time()
        overall_latency = (overall_end_time - overall_start_time) * 1000  # Convert to milliseconds
        print(f"Overall latency (transcription to GPT-4 response): {overall_latency:.2f} ms")

        print("\nPress spacebar to speak again, or 'q' to quit.")
        if keyboard.read_key() == 'q':
            break

if __name__ == "__main__":
    main()