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

import io
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Hugging Face inference endpoint URL
HUGGINGFACE_ENDPOINT = "https://mrholocg8pxhkacd.us-east-1.aws.endpoints.huggingface.cloud"

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
    audio_queue.queue.clear()  # Clear the queue before recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            if keyboard.is_pressed('space'):
                print("Recording... Release spacebar to stop.")
                while keyboard.is_pressed('space'):
                    sd.sleep(100)
                return np.concatenate(list(audio_queue.queue))

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
        start_time = time.time()
        response = requests.post(HUGGINGFACE_ENDPOINT, headers=headers, data=flac_data)
        end_time = time.time()
        response.raise_for_status()
        response_json = response.json()
        transcription = response_json.get("text", None)
        if transcription is None:
            raise ValueError("Transcription key 'text' not found in response.")
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Transcription API latency: {latency:.2f} ms")
        return transcription
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error: {e}")
        if response.text:
            print(f"Response content: {response.text}")
        return None

def get_gpt4_response(transcription):
    """Get response from GPT-4 using OpenAI's API."""
    if transcription is None:
        return "I'm sorry, I couldn't transcribe the audio. Could you please try again?"
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcription}
        ]
    )
    end_time = time.time()
    gpt4_response = response.choices[0].message.content
    
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"GPT-4 API latency: {latency:.2f} ms")
    
    return gpt4_response

def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS API."""
    try:
        start_time = time.time()
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        end_time = time.time()
        
        # Save the audio to a BytesIO object
        audio_data = io.BytesIO(response.content)
        
        # Load the audio using pydub
        audio = AudioSegment.from_mp3(audio_data)
        
        # Play the audio
        play(audio)
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Text-to-Speech API latency: {latency:.2f} ms")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

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

        # Convert GPT-4 response to speech
        print("Converting response to speech...")
        text_to_speech(gpt4_response)

        # Calculate and print overall latency
        overall_end_time = time.time()
        overall_latency = (overall_end_time - overall_start_time) * 1000  # Convert to milliseconds
        print(f"Overall latency (transcription to speech output): {overall_latency:.2f} ms")

        print("\nPress spacebar to speak again, or 'q' to quit.")
        if keyboard.read_key() == 'q':
            break

if __name__ == "__main__":
    main()