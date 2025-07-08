import sounddevice as sd
import numpy as np
import re
from utils.frontend import WavFrontend
from sensevoice_rknn import SenseVoiceInferenceSession, languages
from utils.fsmn_vad import FSMNVad

SAMPLERATE = 16000
BLOCK_DURATION = 1.0  # 1秒
FRAMES_PER_BLOCK = int(SAMPLERATE * BLOCK_DURATION)

frontend = WavFrontend("am.mvn")
model = SenseVoiceInferenceSession(
    "embedding.npy",
    "sense-voice-encoder.rknn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    -1, 4
)
vad = FSMNVad(".")

accumulated_text = ""      # 当前累计输出
collecting = False         # 正在说话的状态
silence_ms = 0             # 静音累计时长（毫秒）

def remove_tags(text):
    # 去除 <|xxx|> 之类标签
    return re.sub(r"<\|.*?\|>", "", text).strip()

def append_new_text(history, new_text):
    # 增量补全，避免重复
    if not history:
        return new_text
    if new_text.startswith(history):
        return new_text
    for i in range(len(history)):
        if new_text.startswith(history[i:]):
            return history + new_text[len(history[i:]):]
    return history + new_text

# -------- 你只需要修改下面这一段即可 --------
# 强制所有输入都当作“中文”或“英文”识别，设定此变量即可
FORCE_LANG = "zh"  # 或 "en"，只用中文模型就写"zh"，只用英文写"en"

def process_block(indata, frames, time, status):
    global accumulated_text, collecting, silence_ms
    mono = indata[:, 0]
    segments = vad.segments_offline(mono)

    if segments:
        # 说话中
        collecting = True
        silence_ms = 0
        feats = frontend.get_features(mono)
        # 只用FORCE_LANG指定的语言识别，始终带标点预测
        raw_text = model(feats[None, ...], language=languages[FORCE_LANG], use_itn=True)
        clean_text = remove_tags(raw_text)
        if clean_text:
            # 增量拼接
            new_accum = append_new_text(accumulated_text, clean_text)
            if new_accum != accumulated_text:
                accumulated_text = new_accum
                print('\r累计内容: ' + accumulated_text + ' ' * 10, end="", flush=True)
    else:
        # 静音
        if collecting:
            silence_ms += int(BLOCK_DURATION * 1000)
            if silence_ms >= 1000:
                # 静音达到1秒，分句
                print("\n[分句输出]:", accumulated_text)
                accumulated_text = ""
                collecting = False
                silence_ms = 0

if __name__ == "__main__":
    print("流式语音识别（每1秒独立识别+静音断句），按Ctrl+C退出。")
    print(f"所有输入都当作 {FORCE_LANG} 识别（带标点预测）")
    with sd.InputStream(
        samplerate=SAMPLERATE, channels=1, dtype='float32',
        blocksize=FRAMES_PER_BLOCK, callback=process_block
    ):
        while True:
            sd.sleep(100)
