"""Decode two separately logged transmissions from one continuous phone video."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from communication import decode,frame_bits
from heart_rate import read_video


def decode_trace(t,y,configs,split_seconds,split_assignment='Video midpoint split chosen before decoding'):
    if len(configs)!=2:raise ValueError('Exactly two transmitter records are required.')
    windows=[(float(t[0]),split_seconds),(split_seconds,float(t[-1])+np.median(np.diff(t)))]
    decoded=[]
    for config,(begin,end) in zip(configs,windows):
        mask=(t>=begin)&(t<end)
        if mask.sum()<3:raise ValueError('Empty analysis segment.')
        result=decode(t[mask],y[mask],config)
        result.update(segment_seconds=[begin,end],
                      full_packet_within_segment=bool(result['start_seconds']+result['packet_duration_seconds']<=end+1e-6),
                      segment_assignment=f'Chronological transmitter record order; {split_assignment}. Synchronization uses public preamble only.')
        decoded.append((result,mask))
    return decoded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('manifests',nargs=2,type=Path)
    p.add_argument('--roi',nargs=4,type=int,required=True)
    p.add_argument('--out-root',type=Path,default=Path('results/communication'))
    p.add_argument('--split-seconds',type=float,help='Optional boundary selected from the recording before decoding.')
    p.add_argument('--review-name',help='Separate review directory name for an additional continuous recording.')
    args=p.parse_args()
    pairs=[(path,json.loads(path.read_text(encoding='utf-8'))) for path in args.manifests]
    pairs.sort(key=lambda row:datetime.fromisoformat(row[1]['created_at'].replace('Z','+00:00')))
    for path,config in pairs:
        if config.get('status')!='completed':raise ValueError(f'Not a completed transmission: {path}')
        if config['bits']!=frame_bits(config['message']):raise ValueError(f'Invalid protocol bits: {path}')
    if pairs[0][1]['distance_cm']!=pairs[1][1]['distance_cm']:raise ValueError('Distance must remain the same within this recording.')
    t,values,fs,timing=read_video(args.video,args.roi)
    split=float((t[0]+t[-1]+1/fs)/2) if args.split_seconds is None else args.split_seconds
    assignment='Video midpoint split chosen before decoding' if args.split_seconds is None else 'Explicit video-time split chosen before decoding'
    sha=hashlib.sha256(args.video.read_bytes()).hexdigest()
    decoded=decode_trace(t,values[:,0],[row[1] for row in pairs],split,assignment)
    summary=[]
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for ax,(path,config),(result,mask) in zip(axes,pairs,decoded):
        distance=config['distance_cm'];alpha=config['delta_alpha'];T=config['symbol_seconds']
        folder=f'd{distance}_a{str(alpha).replace(".","p")}_t{T}_r1'
        out=args.out_root/folder;out.mkdir(parents=True,exist_ok=True)
        result.update(video=str(args.video.resolve()),video_sha256=sha,manifest=str(path.resolve()),
                      roi=args.roi,sampling_hz=fs,timing=timing,
                      recorded_receive_mode='Phone video decoded offline; real-time phone receiver not demonstrated.',
                      roi_selection='Fixed image-interior region selected by frame inspection before decoding; same ROI for both transmissions in this video.')
        (out/'result.json').write_text(json.dumps(result,indent=2))
        with (out/'intensity.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','mean_gray']);writer.writerows(zip(t[mask],values[mask,0]))
        title=f'{distance} cm, alpha difference {alpha}: {result["message_decoded"]!r}, CRC {result["crc_ok"]}, BER {result["ber"]:.3f}'
        ax.plot(t[mask],values[mask,0],lw=.8);ax.axvline(result['start_seconds'],color='r',ls='--')
        ax.set(xlabel='Time from continuous-video start (s)',ylabel='ROI mean intensity',title=title)
        single,sax=plt.subplots(figsize=(11,3.5));sax.plot(t[mask],values[mask,0],lw=.8);sax.axvline(result['start_seconds'],color='r',ls='--')
        sax.set(xlabel='Time from continuous-video start (s)',ylabel='ROI mean intensity',title=title)
        single.tight_layout();single.savefig(out/'waveform.png',dpi=170);plt.close(single)
        summary.append({k:result[k] for k in ['distance_cm','delta_alpha','nominal_raw_bps','symbol_seconds','start_seconds','segment_seconds','full_packet_within_segment','message_decoded','crc_ok','total_transmitted_bits','correct_bits','ber','goodput_bps','payload_goodput_bps']})
    fig.tight_layout()
    combined=args.out_root/(args.review_name or f'd{pairs[0][1]["distance_cm"]}_continuous_review')
    combined.mkdir(parents=True,exist_ok=True)
    fig.savefig(combined/'both_transmissions.png',dpi=170);plt.close(fig)
    (combined/'summary.json').write_text(json.dumps({'source_video':str(args.video.resolve()),'video_sha256':sha,'roi':args.roi,'split_seconds':split,'transmissions':summary},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
