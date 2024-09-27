import os
import csv
from pyannote.audio import Pipeline
import torch
import torchaudio
import subprocess
import whisper

# 加載 Whisper Large 模型，啟用自動語言偵測
whisper_model = whisper.load_model("large")  # 使用 large 模型，自動語言偵測

# 加載 Pyannote 預訓練模型
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your_api_token"
)

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU instead.")
    device = torch.device("cpu")

# 定義資料夾路徑
input_folder = r"D:\Project\test"  # 修改為你存放 .wav 檔案的資料夾路徑
output_folder = r"D:\Project\test\output"  # 修改為你希望存放結果的資料夾路徑

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 使用 ffmpeg 進行轉檔的函數
def convert_to_wav(input_file, output_file):
    try:
        command = f"ffmpeg -i \"{input_file}\" -acodec pcm_s16le -ar 16000 \"{output_file}\""
        subprocess.run(command, shell=True, check=True)
        print(f"Converted {input_file} to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {input_file}: {e}")
        return False

# 搜尋指定資料夾中的所有檔案，並確保只處理檔案
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 將 Whisper 和 Pyannote 時間段整合
def match_speaker_to_transcript(transcription_segments, diarization):
    result = []
    
    for segment in transcription_segments:
        whisper_start = segment["start"]
        whisper_end = segment["end"]
        text = segment["text"]

        # 記錄說話者出現次數的字典
        speaker_counts = {}

        # 遍歷 Pyannote 的說話者分段，檢查是否與 Whisper 的時間段重疊
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(whisper_start, turn.start)
            overlap_end = min(whisper_end, turn.end)

            # 如果有重疊，增加該說話者的計數
            if overlap_start < overlap_end:
                if speaker not in speaker_counts:
                    speaker_counts[speaker] = 0
                speaker_counts[speaker] += (overlap_end - overlap_start)

        # 找出出現最多的說話者
        if speaker_counts:
            most_common_speaker = max(speaker_counts, key=speaker_counts.get)
            result.append({
                "start": whisper_start,
                "end": whisper_end,
                "text": text,
                "speaker": most_common_speaker
            })
        else:
            result.append({
                "start": whisper_start,
                "end": whisper_end,
                "text": text,
                "speaker": "Unknown"
            })

    return result

# 對每個檔案進行處理
for file in files:
    input_path = os.path.join(input_folder, file)
    
    # 如果是 .wav 檔案，嘗試載入，檢查是否損壞
    if file.endswith('.wav'):
        try:
            # 嘗試加載音訊檔案
            waveform, sample_rate = torchaudio.load(input_path)
            print(f"Processing {file} as a valid wav file.")
        except Exception as e:
            print(f"Error loading {file}: {e}. Attempting to convert.")
            # 如果無法加載，嘗試轉換為新的 wav
            new_wav_path = os.path.join(input_folder, f"{os.path.splitext(file)[0]}_converted.wav")
            if convert_to_wav(input_path, new_wav_path):
                input_path = new_wav_path  # 更新路徑，指向轉換後的檔案
            else:
                continue  # 如果轉換失敗，跳過該檔案
    else:
        # 如果不是 .wav 檔案，直接轉檔
        new_wav_path = os.path.join(input_folder, f"{os.path.splitext(file)[0]}.wav")
        if convert_to_wav(input_path, new_wav_path):
            input_path = new_wav_path  # 更新路徑，指向轉換後的檔案
        else:
            continue  # 如果轉換失敗，跳過該檔案

    # 使用 Whisper 進行語音轉文字，語言自動偵測
    result = whisper_model.transcribe(input_path)
    transcription_segments = result["segments"]
    
    # 使用 Pyannote 進行說話者分段處理
    diarization = pipeline(input_path)
    
    # 將 Whisper 和 Pyannote 的結果進行整合
    matched_result = match_speaker_to_transcript(transcription_segments, diarization)

    # 設定輸出 CSV 檔案路徑
    output_csv_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_speaker_transcription.csv")
    
    # 將結果儲存到 CSV 檔案中
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["start", "end", "text", "speaker"])
        writer.writeheader()
        for row in matched_result:
            writer.writerow(row)
    
    print(f"Finished processing {file}, results saved to {output_csv_file}")
