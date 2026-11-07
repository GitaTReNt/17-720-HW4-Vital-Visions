"""Recover a BFSK packet without inventing a missing transmitter log."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from communication import PREAMBLE,crc8,symbols_at
from heart_rate import read_video


def receive(t,y,T,f0,f1):
    fs=1/np.median(np.diff(t))
    if f1>=fs/2:raise ValueError('Camera sampling violates Nyquist for the configured tones.')
    max_start=t[-1]-len(PREAMBLE)*T
    if max_start<t[0]:raise ValueError('Recording shorter than preamble.')
    step=max(1/fs,T/12);best=None
    def score(start):
        bits,conf=symbols_at(t,y,start,len(PREAMBLE),T,f0,f1)
        match=float(np.mean(bits==PREAMBLE))
        return (match+.12*float(np.mean(conf)),float(start),match)
    for start in np.arange(t[0],max_start+1e-8,step):
        candidate=score(start)
        if best is None or candidate[0]>best[0]:best=candidate
    coarse=best[1]
    for start in np.arange(max(t[0],coarse-step),min(max_start,coarse+step)+1e-8,1/(4*fs)):
        candidate=score(start)
        if candidate[0]>best[0]:best=candidate
    result={'synchronized':bool(best[2]>=12/13),'start_seconds':best[1],
            'preamble_correct_fraction':best[2],'message_decoded':None,'crc_ok':False,
            'received_bits':[],'payload_length_bytes':None}
    if not result['synchronized']:return result
    header,_=symbols_at(t,y,best[1]+len(PREAMBLE)*T,8,T,f0,f1)
    if np.any(header<0):return result|{'error':'Header contains missing symbols.'}
    length=int(''.join(map(str,header)),2)
    if not 1<=length<=32:return result|{'error':'Decoded length is outside the protocol range.'}
    count=len(PREAMBLE)+8*(length+2)
    bits,confidence=symbols_at(t,y,best[1],count,T,f0,f1)
    result.update(payload_length_bytes=length,received_bits=bits.tolist(),confidence=confidence.tolist(),
                  decoded_frame_bits=count,packet_duration_seconds=count*T)
    body=bits[len(PREAMBLE):]
    if np.any(body<0):return result|{'error':'Packet contains missing symbols.'}
    raw=bytes(int(''.join(map(str,body[i:i+8])),2) for i in range(0,len(body),8))
    result.update(message_decoded=raw[1:-1].decode('utf-8',errors='replace'),
                  crc_ok=bool(crc8(raw[:-1])==raw[-1]),received_body_bytes=list(raw))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('--roi',nargs=4,type=int,required=True)
    p.add_argument('--symbol-seconds',type=float,required=True)
    p.add_argument('--delta-alpha',type=float,required=True)
    p.add_argument('--distance-cm',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.symbol_seconds<=0:p.error('Symbol duration must be positive.')
    t,values,fs,timing=read_video(args.video,args.roi)
    result=receive(t,values[:,0],args.symbol_seconds,2/args.symbol_seconds,4/args.symbol_seconds)
    result.update(status='recorded_video_without_transmitter_log',transmitter_log_available=False,
                  video=str(args.video.resolve()),video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                  sampling_hz=fs,timing=timing,roi=args.roi,symbol_seconds=args.symbol_seconds,
                  f0_hz=2/args.symbol_seconds,f1_hz=4/args.symbol_seconds,
                  nominal_raw_bps=1/args.symbol_seconds,delta_alpha=args.delta_alpha,distance_cm=args.distance_cm,
                  parameter_source='Earlier experiment setup and filename; no original transmitter timing log exists.',
                  decoding_method='Public preamble synchronization; payload length, message and CRC recovered from video without expected payload.',
                  ber=None,goodput_bps=None,
                  comparison_note='No bit-error rate assigned by comparing the decoded message to itself. A known, independently specified transmitted message is needed for BER.')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    with (args.out/'intensity.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','mean_gray']);w.writerows(zip(t,values[:,0]))
    fig,ax=plt.subplots(figsize=(11,3.5));ax.plot(t,values[:,0],lw=.8)
    ax.axvline(result['start_seconds'],color='r',ls='--')
    ax.set(xlabel='Recording time (s)',ylabel='ROI mean intensity',title=f"Without TX log: recovered {result['message_decoded']!r}; CRC {'pass' if result['crc_ok'] else 'fail'}")
    fig.tight_layout();fig.savefig(args.out/'waveform.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['received_bits','confidence']},indent=2))


if __name__=='__main__':main()
