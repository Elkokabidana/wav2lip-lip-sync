import cv2
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
#____________
import torch
#_____________
import subprocess
from tempfile import TemporaryDirectory

# تنظیمات
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 2.0
BUFFER_SIZE = 3
WAV2LIP_MODEL_PATH = "checkpoints/wav2lip.pth"  # مسیر مدل Wav2Lip
OUTPUT_VIDEO_PATH = "../output.mp4"
TEMP_VIDEO_PATH = "../temp_video.mp4"
TEMP_AUDIO_PATH = "temp_audio.wav"
#___________________________________
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 در حال استفاده از: {device.upper()}")
#__________________________________
# صف‌ها برای تبادل داده
video_queue = queue.Queue(maxsize=BUFFER_SIZE)
audio_queue = queue.Queue(maxsize=BUFFER_SIZE)
output_queue = queue.Queue(maxsize=BUFFER_SIZE)

# متغیرهای سراسری
running = True
processing_ready = threading.Event()
audio_chunks = []  # برای ذخیره موقت داده‌های صوتی
# ایجاد پوشه موقت برای فایل‌های میانی
temp_dir = TemporaryDirectory()
temp_path = temp_dir.name
def record_audio():
    """ضبط صدا در چانک‌ها و افزودن به صف و ذخیره موقت"""
    global running
    print("📢 شروع ضبط صدا...")

    chunk_size = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION)

    def audio_callback(indata, frames, time, status):
        if running:
            audio_data = indata.copy()
            audio_chunks.append(audio_data)  # ذخیره برای فایل موقت
            if not audio_queue.full():
                audio_queue.put(audio_data)
                processing_ready.set()

    try:
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1,
                            callback=audio_callback, blocksize=chunk_size):
            print("✅ ضبط صدا آغاز شد.")
            while running:
                time.sleep(0.1)
    except Exception as e:
        print(f"❌ خطا در ضبط صدا: {e}")
        running = False

def capture_video():
    """ضبط فریم‌های ویدیو و افزودن به صف"""
    global running
    print("🎥 آماده‌سازی دوربین...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    if not cap.isOpened():
        print("❌ دوربین یافت نشد!")
        running = False
        return

    print("✅ دوربین آماده است.")
    frame_interval = 1.0 / 30  # 30 فریم بر ثانیه
    last_frame_time = time.time()

    try:
        while running:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ خطا در ضبط فریم.")
                    continue
                if not video_queue.full():
                    video_queue.put(frame)
                    processing_ready.set()
                last_frame_time = current_time
                cv2.imshow('ورودی ویدئو', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
    except Exception as e:
        print(f"❌ خطا در ضبط ویدیو: {e}")
        running = False
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_wav2lip():
    """پردازش صدا و ویدیو با Wav2Lip و ذخیره فریم‌های پردازش‌شده"""
    global running
    print("🧠 شروع پردازشگر Wav2Lip...")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame_count = 0

    while running:
        processing_ready.wait(timeout=1.0)
        processing_ready.clear()

        if video_queue.empty() or audio_queue.empty():
            continue

        frame = video_queue.get()
        audio_data = audio_queue.get()

        frame_path = os.path.join(temp_path, f"frame_{frame_count}.jpg")
        audio_path = os.path.join(temp_path, f"audio_{frame_count}.wav")
        output_path = os.path.join(temp_path, f"output_{frame_count}.mp4")

        # تشخیص چهره و برش
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin = int(w * 0.1)
            x, y = max(0, x - margin), max(0, y - margin)
            w, h = min(frame.shape[1] - x, w + 2 * margin), min(frame.shape[0] - y, h + 2 * margin)
            face_frame = frame[y:y + h, x:x + w]
            cv2.imwrite(frame_path, face_frame)
        else:
            cv2.imwrite(frame_path, frame)

        # ذخیره چانک صدا
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        sf.write(audio_path, audio_data_int16, AUDIO_SAMPLE_RATE, subtype='PCM_16')

        # اجرای Wav2Lip
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inference_path = os.path.join(script_dir, "inference.py")
        command = [
            "python", inference_path,
            "--checkpoint_path", WAV2LIP_MODEL_PATH,
            "--face", frame_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--pads", "0", "0", "0", "0",
            "--nosmooth",
            "--resize_factor", "1",
            "--fps", "30"
        ]

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, check=True, timeout=1.5 * AUDIO_CHUNK_DURATION)
            print(f"Wav2Lip stdout: {result.stdout}")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                output_cap = cv2.VideoCapture(output_path)
                ret, output_frame = output_cap.read()
                output_cap.release()
                if ret:
                    output_queue.put(output_frame)
                    print(f"✅ فریم {frame_count} پردازش شد")
                else:
                    print(f"⚠ خطا در خواندن خروجی برای فریم {frame_count}")
            else:
                print(f"⚠ خروجی برای فریم {frame_count} تولید نشد")

        except subprocess.CalledProcessError as e:
            print(f"خطای Wav2Lip: {e.stderr}")
        except subprocess.TimeoutExpired:
            print(f"تایم‌اوت پردازش فریم {frame_count}")
        except Exception as e:
            print(f"خطای پردازش: {e}")

        # پاکسازی فایل‌های موقت
        for file_path in [frame_path, audio_path, output_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        frame_count += 1

def display_output():
    """نمایش و ذخیره فریم‌های پردازش‌شده در فایل موقت"""
    global running
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, 30.0, (WEBCAM_WIDTH, WEBCAM_HEIGHT))

    while running:
        if not output_queue.empty():
            output_frame = output_queue.get()
            out.write(output_frame)  # ذخیره فریم در فایل موقت
            cv2.imshow('خروجی Wav2Lip', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    out.release()
    cv2.destroyAllWindows()

def main():
    """تابع اصلی برای هماهنگی و ادغام با FFmpeg"""
    global running
    try:
        # شروع threadها
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

        # ذخیره صدا در فایل موقت
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
            sf.write(TEMP_AUDIO_PATH, audio_data, AUDIO_SAMPLE_RATE)

        # ادغام ویدئو و صدا با FFmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", TEMP_VIDEO_PATH,
            "-i", TEMP_AUDIO_PATH,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-y",
            OUTPUT_VIDEO_PATH
        ]
        subprocess.run(ffmpeg_cmd)

        # حذف فایل‌های موقت
        os.remove(TEMP_VIDEO_PATH)
        os.remove(TEMP_AUDIO_PATH)

    except KeyboardInterrupt:
        print("👋 برنامه متوقف شد")
    finally:
        running = False
        temp_dir.cleanup()
        cv2.destroyAllWindows()
        print("🧹 پاکسازی انجام شد")

if __name__ == "__main__":
    main()