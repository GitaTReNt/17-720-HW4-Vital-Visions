"""Extract camera PPG and estimate heart rate from a real finger video."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def read_video(path, roi=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Missing frame rate; re-encode the video with valid timing.')
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if roi:
            x, y, w, h = roi
            if min(x, y) < 0 or min(w, h) <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
                raise ValueError('ROI lies outside the video frame.')
            frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, g, r = frame.mean(axis=(0, 1))
        rows.append([timestamp, float(gray.mean()), r, g, b,
                     float(np.mean((gray <= 1) | (gray >= 254)))])
    cap.release()
    if len(rows) < 3:
        raise ValueError('Video contains fewer than 3 readable frames.')
    data = np.asarray(rows)
    measured = data[:, 0]
    if np.all(np.diff(measured) > 0):
        timing = 'decoded frame timestamps, resampled to uniform grid'
        fs = 1 / np.median(np.diff(measured))
        t = np.arange(int(np.floor((measured[-1]-measured[0])*fs))+1) / fs
        values = np.column_stack([np.interp(t+measured[0], measured, data[:, j])
                                  for j in range(1, data.shape[1])])
    else:
        timing = 'nominal FPS fallback; variable-frame-rate timing not verified'
        fs, t, values = fps, np.arange(len(data))/fps, data[:, 1:]
    return t, values, fs, timing


def estimate(t, x, fs):
    if len(x)/fs < 10:
        raise ValueError('Need at least 10 seconds after trimming; record 30 seconds.')
    if np.std(x) < 0.02:
        raise ValueError('Almost no intensity variation; inspect camera coverage and exposure.')
    centered = x - np.mean(x)
    filtered = signal.sosfiltfilt(signal.butter(3, 0.5, 'highpass', fs=fs, output='sos'), centered)
    # Hann window suppresses leakage; no zero padding disguises the true resolution.
    freq = np.fft.rfftfreq(len(filtered), 1/fs)
    amplitude = np.abs(np.fft.rfft(filtered*np.hanning(len(filtered)))) * 2/np.hanning(len(filtered)).sum()
    band = (freq >= 1) & (freq <= 100/60)
    if not band.any():
        raise ValueError('No FFT bins in the assignment search range.')
    k = np.flatnonzero(band)[np.argmax(amplitude[band])]
    f_peak = float(freq[k])
    samples_per_beat = fs/f_peak
    # Allow timing variation while suppressing multiple detections within one beat.
    min_distance = max(1, int(np.floor(0.65*samples_per_beat)))
    prominence = max(0.15*float(np.std(filtered)), 1e-6)
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    intervals = np.diff(t[peaks])
    hr_time = float(60/np.mean(intervals)) if len(intervals) else None
    warnings = []
    if k in [np.flatnonzero(band)[0], np.flatnonzero(band)[-1]]:
        warnings.append('FFT maximum lies at the 60-100 bpm search boundary; true HR may be outside this range.')
    if hr_time is None or abs(hr_time-60*f_peak) > 10:
        warnings.append('FFT and peak timing disagree; inspect the waveform and peak selection.')
    return filtered, freq, amplitude, peaks, {
        'fft_hz': f_peak, 'fft_bpm': 60*f_peak, 'time_domain_bpm': hr_time,
        'samples_per_beat': samples_per_beat, 'min_peak_distance_samples': min_distance,
        'peak_prominence': prominence, 'peak_count': len(peaks),
        'fft_resolution_hz': fs/len(x), 'search_range_bpm': [60, 100], 'warnings': warnings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/heart_rate'))
    p.add_argument('--trim-seconds', type=float, default=2)
    p.add_argument('--roi', type=int, nargs=4, metavar=('X','Y','W','H'))
    args = p.parse_args()
    t, values, fs, timing = read_video(args.video, args.roi)
    if args.trim_seconds < 0:
        p.error('--trim-seconds must be nonnegative')
    keep = t >= args.trim_seconds
    t, values = t[keep], values[keep]
    filtered, freq, amp, peaks, result = estimate(t, values[:, 0], fs)
    result.update(status='real_video_analysis', input_file=str(args.video.resolve()),
                  input_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                  sampling_hz=fs, timing=timing, duration_seconds=len(t)/fs,
                  trim_seconds=args.trim_seconds, roi=args.roi,
                  saturated_pixel_fraction_mean=float(values[:, -1].mean()),
                  interpretation='Signal estimate only; no reference-device accuracy claim.')
    if values[:, -1].mean() > 0.2:
        result['warnings'].append('More than 20% of pixels are near black/white clipping on average.')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'signal.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['time_s','gray','red','green','blue','clipped_fraction','filtered','is_peak'])
        mask = np.zeros(len(t), dtype=int); mask[peaks] = 1
        w.writerows(np.column_stack([t, values, filtered, mask]))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(freq, amp); ax.axvspan(1, 100/60, color='gray', alpha=.15)
    ax.axvline(result['fft_hz'], color='crimson', ls='--', label=f"Estimate: {result['fft_hz']:.3f} Hz ({result['fft_bpm']:.1f} bpm)")
    ax.set(xlim=(0, 4), xlabel='Frequency (Hz)', ylabel='Amplitude (gray levels)', title='Camera PPG: windowed FFT')
    ax.legend(); fig.tight_layout(); fig.savefig(args.out/'fft.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t, filtered, lw=1); ax.plot(t[peaks], filtered[peaks], 'ro', ms=4)
    bpm = result['time_domain_bpm']
    ax.set(xlabel='Time (s)', ylabel='High-pass intensity', title=f"Selected heartbeat peaks: {bpm:.1f} bpm" if bpm else 'Insufficient peaks')
    fig.tight_layout(); fig.savefig(args.out/'heartbeat_peaks.png', dpi=180); plt.close(fig)
    (args.out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
