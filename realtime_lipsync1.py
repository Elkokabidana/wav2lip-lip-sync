import cv2
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
import soundfile as sf

# تنظیمات
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
AUDIO_SAMPLE_RATE = 16000
BUFFER_SIZE = 2
WAV2LIP_MODEL_PATH = "checkpoints/wav2lip.pth"

# تعیین دستگاه GPU یا CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 در حال استفاده از: {str(device).upper()}")

# صف‌ها برای داده‌ها
audio_queue = queue.Queue(maxsize=BUFFER_SIZE)
video_queue = queue.Queue(maxsize=BUFFER_SIZE)
output_queue = queue.Queue(maxsize=BUFFER_SIZE)

running = True
processing_ready = threading.Event()

def record_audio():
    def audio_callback(indata, frames, time, status):
        if running and not audio_queue.full():
            audio_queue.put(indata.copy())
            processing_ready.set()
    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=audio_callback):
        print("✅ ضبط صدا آغاز شد.")
        while running:
            time.sleep(0.1)

def capture_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    print("✅ دوربین آماده است.")
    while running:
        ret, frame = cap.read()
        if ret and not video_queue.full():
            video_queue.put(frame)
            processing_ready.set()
        cv2.imshow('ورودی ویدئو', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_wav2lip():
    wav2lip_model = torch.load(WAV2LIP_MODEL_PATH, map_location=device)
    wav2lip_model = wav2lip_model.to(device)
    wav2lip_model.eval()

    while running:
        processing_ready.wait()
        processing_ready.clear()

        if video_queue.empty() or audio_queue.empty():
            continue

        frame = video_queue.get()
        audio_data = audio_queue.get()

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = wav2lip_model(frame_tensor, audio_tensor)

        output_frame = output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_queue.put(output_frame)
        torch.cuda.empty_cache()

def display_output():
    while running:
        if not output_queue.empty():
            output_frame = output_queue.get()
            cv2.imshow('خروجی Wav2Lip', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    threads = [
        threading.Thread(target=record_audio, daemon=True),
        threading.Thread(target=capture_video, daemon=True),
        threading.Thread(target=process_wav2lip, daemon=True),
        threading.Thread(target=display_output, daemon=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()