import numpy as np

class FSMNVad:
    def __init__(self, model_dir=None):
        self.energy_threshold = 0.001
        self.min_speech_ms = 200
        self.sample_rate = 16000

    def segments_offline(self, audio):
        win_size = int(0.03 * self.sample_rate)
        hop_size = int(0.01 * self.sample_rate)

        marks = []
        for i in range(0, len(audio) - win_size, hop_size):
            win = audio[i:i+win_size]
            marks.append(1 if np.mean(np.abs(win)) > self.energy_threshold else 0)

        # 找到连续有声片段
        segments = []
        in_speech = False
        seg_start = 0
        for idx, mark in enumerate(marks):
            if mark and not in_speech:
                in_speech = True
                seg_start = idx * hop_size
            elif not mark and in_speech:
                in_speech = False
                seg_end = idx * hop_size
                if seg_end - seg_start > self.min_speech_ms / 1000 * self.sample_rate:
                    segments.append((seg_start, seg_end))
        if in_speech:
            seg_end = len(marks) * hop_size
            if seg_end - seg_start > self.min_speech_ms / 1000 * self.sample_rate:
                segments.append((seg_start, seg_end))
        # print("VAD分析: ", segments)
        return segments
