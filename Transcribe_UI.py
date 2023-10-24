import tkinter as tk
from tkinter import filedialog, messagebox
import platform
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from resemblyzer import preprocess_wav, VoiceEncoder,sampling_rate
from spectralcluster import SpectralClusterer
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import AudioFileClip
import torch
import soundfile as sf
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import warnings
import time
import json
from resemblyzer import sampling_rate

app = tk.Tk()
app.title("Video Transcription Tool")
app.geometry('400x450')  # set window size
app.configure(bg='#e0e0e0')  # Set a background color

warnings.simplefilter("ignore")
# model_directory = r"models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
encoder=VoiceEncoder()

def select_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*")])
    video_path_var.set(filepath)
def create_labelling(labels, wav_splits):
    
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i-1]:
            temp = [str(labels[i-1]), start_time, time]
            labelling.append(tuple(temp))
            start_time = time
        if i == len(times) - 1:
            temp = [str(labels[i]), start_time, time]
            labelling.append(tuple(temp))

    return labelling
def process_audio(filename):
    # Your existing processing code
    wav = preprocess_wav(filename)
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    clusterer = SpectralClusterer(min_clusters=2, max_clusters=100)
    labels = clusterer.predict(cont_embeds)
    labelling = create_labelling(labels, wav_splits)

    transcriptions = []

    for speaker, start, end in labelling:
        segment = wav[int(start * sampling_rate):int(end * sampling_rate)]
        input_values = processor(segment, return_tensors="pt", padding=True).input_values
        input_values = input_values.float()
        input_values = input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        transcriptions.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "transcription": transcription
        })

    return transcriptions

def process_transcription():
    video_path = video_path_var.get()

    # Check if video path is not empty
    if not video_path:
        messagebox.showerror("Error", "Please select a video file")
        return
    
    # Your transcription logic here...
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile("audio.wav")
    # wav = preprocess_wav("audio.wav")
    audio = AudioSegment.from_wav("audio.wav")
    half_duration = len(audio) // 2
    part1 = audio[:half_duration]
    part2 = audio[half_duration:]
    print('Audio written')
    # a= time.time()
    #... [rest of your transcription code]

    part1.export("part1.wav", format="wav")
    part2.export("part2.wav", format="wav")

    # Process each part separately
    a=time.time()
    transcriptions_part1 = process_audio("part1.wav")
    print('First part done')
    transcriptions_part2 = process_audio("part2.wav")
    b=time.time()
    for segment in transcriptions_part2:
        segment["start"] += half_duration / 1000  # pydub works in milliseconds
        segment["end"] += half_duration / 1000

    # Merge the results
    merged_transcriptions = transcriptions_part1 + transcriptions_part2

    # Save the merged results to a JSON file
    with open('transcriptions.json', 'w') as json_file:
        json.dump(merged_transcriptions, json_file, indent=4)

    # Optionally, you can remove the temporary files
    os.remove("part1.wav")
    os.remove("part2.wav")
    messagebox.showinfo("Info", "Transcription Completed and saved to transcriptions.json in {} seconds".format(b-a))

    # Once transcription is complete, show a message box
# b=time.time()

# Layout
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system_info = f"""
OS: {platform.system()}
OS Version: {platform.version()}
Machine: {platform.machine()}
Processor: {platform.processor()}
PyTorch Device: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}
"""

# Layout
tk.Label(app, text="System Details", font=("Arial", 16)).pack(pady=10)
tk.Label(app, text=system_info, justify=tk.LEFT).pack(pady=10)
tk.Label(app, text="Select Video for Transcription").pack(pady=10)
video_path_var = tk.StringVar()
tk.Entry(app, textvariable=video_path_var, width=40).pack(pady=5)
tk.Button(app, text="Browse", command=select_video).pack(pady=5)
tk.Button(app, text="Start Transcription", command=process_transcription).pack(pady=20)

app.mainloop()
