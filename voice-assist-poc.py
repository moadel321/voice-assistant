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
from pydub import AudioSegment
import csv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set the path to the FFmpeg executables
ffmpeg_path = r"C:\ProgramData\chocolatey\lib\ffmpeg-full\tools\ffmpeg\bin"  # Replace with your actual path
os.environ["PATH"] += os.pathsep + ffmpeg_path

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
def get_gpt4_response(transcription):
    """Get response from GPT-4 using OpenAI's API."""
    if transcription is None:
        return "I'm sorry, I couldn't transcribe the audio. Could you please try again?"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcription}
        ]
    )
    
    return response.choices[0].message.content

@measure_latency
def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS API and play it directly."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Load MP3 data into an AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(response.content))
        
        # Convert to raw PCM audio data
        audio_data = np.array(audio_segment.get_array_of_samples())
        
        # Play the audio
        sd.play(audio_data, samplerate=audio_segment.frame_rate)
        sd.wait()  # Wait until the audio is finished playing
        
        return "Audio played successfully"
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None

def write_to_csv(data, filename="latency_log.csv"):
    fieldnames = ["timestamp", "turn", "transcription_latency", "llm_latency", "tts_latency", "total_latency", "highest_latency"]
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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

        # Get GPT-4 response
        gpt4_response, llm_latency = get_gpt4_response(transcription)
        print(f"GPT-4 Response: {display_arabic(gpt4_response)}")

        # Convert GPT-4 response to speech
        print("Converting response to speech...")
        tts_result, tts_latency = text_to_speech(gpt4_response)

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