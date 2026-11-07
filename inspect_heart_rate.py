"""Inspect sampled frames and the extracted intensity trace without changing it."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import cv2
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

root=Path(__file__).resolve().parent
out=root/'results/heart_rate/IMG_2324';out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(root/'data/heart_rate/IMG_2324.MOV'))
fps=cap.get(cv2.CAP_PROP_FPS);n=cap.get(cv2.CAP_PROP_FRAME_COUNT)
meta={'nominal_fps':fps,'nominal_frames':n,'nominal_duration_seconds':n/fps,'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
sheet=Image.new('RGB',(5*220,420),'#e7edf2');draw=ImageDraw.Draw(sheet);rows=[]
for k,seconds in enumerate([0,2,10,20,30]):
    cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,frame=cap.read()
    if not ok:continue
    row={'seconds_requested':seconds,'timestamp_seconds':cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
         'mean_RGB':frame.mean(axis=(0,1))[::-1].tolist(),
         'saturated_RGB_fraction':((frame>=254).mean(axis=(0,1))[::-1]).tolist()}
    rows.append(row);im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((200,370))
    sheet.paste(im,(k*220+10,30));draw.text((k*220+10,8),f'{seconds} s',fill='#172d3d')
cap.release();sheet.save(out/'sampled_frames.png')
meta['sampled_frames']=rows
(out/'video_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
trace=out/'signal.csv'
if trace.exists():
    data=np.genfromtxt(trace,delimiter=',',names=True)
    t=data['time_s'];gray=data['gray'];filtered=data['filtered'];fs=1/np.median(np.diff(t))
    fig,axes=plt.subplots(3,1,figsize=(11,8))
    axes[0].plot(t,gray);axes[0].set(ylabel='Mean grayscale',title='Real video: brightness trace and quality review')
    for channel in ['red','green','blue']:axes[1].plot(t,data[channel],label=channel)
    axes[1].set(ylabel='Channel mean');axes[1].legend(ncol=3)
    axes[2].plot(t,filtered,lw=.9);axes[2].set(xlabel='Time from video start (s)',ylabel='High-pass grayscale')
    fig.tight_layout();fig.savefig(out/'quality_review.png',dpi=170);plt.close(fig)
    windows=[]
    for start in [2,12,22]:
        mask=(t>=start)&(t<start+10)
        if mask.sum()<8*fs:continue
        freq,power=welch(filtered[mask],fs=fs,nperseg=mask.sum())
        band=(freq>=1-1e-9)&(freq<=100/60+1e-9);k=np.flatnonzero(band)[np.argmax(power[band])]
        windows.append({'start_seconds':start,'end_seconds':min(start+10,float(t[-1])),
                        'dominant_frequency_1_to_1p667_hz':float(freq[k]),'bpm':float(60*freq[k])})
    diagnostic={'status':'quality_review','gray_min_max':[float(gray.min()),float(gray.max())],
                'gray_std':float(gray.std()),'windows':windows,
                'note':'Window FFT values have coarse 0.1 Hz resolution; these are stability checks, not independent reference measurements.'}
    (out/'quality_review.json').write_text(json.dumps(diagnostic,indent=2));print(json.dumps(diagnostic,indent=2))
