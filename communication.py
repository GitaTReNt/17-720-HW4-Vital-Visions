"""Decode screen alpha BFSK from a phone recording and a transmitter manifest."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from heart_rate import read_video

PREAMBLE = [1,1,1,1,1,0,0,1,1,0,1,0,1]


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 7) & 255 if crc & 128 else (crc << 1) & 255
    return crc


def byte_bits(data):
    return [(b >> shift) & 1 for b in data for shift in range(7, -1, -1)]


def frame_bits(message):
    data = message.encode('utf-8')
    if not 1 <= len(data) <= 32:
        raise ValueError('Message must have 1-32 UTF-8 bytes.')
    return PREAMBLE + byte_bits(bytes([len(data)]) + data + bytes([crc8(bytes([len(data)])+data)]))


def tone_score(t, y, f):
    # Regression removes local illumination offset/drift and supports irregular timing.
    tt = t - t[0]
    X = np.column_stack([np.ones(len(t)), tt, np.sin(2*np.pi*f*tt), np.cos(2*np.pi*f*tt)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    null = np.column_stack([np.ones(len(t)),tt])
    base, *_ = np.linalg.lstsq(null,y,rcond=None)
    residual0 = float(np.sum((y-null@base)**2))
    residual1 = float(np.sum((y-X@coef)**2))
    return max(0., (residual0-residual1)/(residual0+1e-12))


def symbols_at(t, y, start, count, symbol, f0, f1, margin=0.08):
    bits=[]; confidence=[]
    step = np.median(np.diff(t))
    for k in range(count):
        a=start+(k+margin)*symbol; b=start+(k+1-margin)*symbol
        mask=(t>=a)&(t<b)
        if mask.sum()<6 or a < t[0]-step or b > t[-1]+step:
            bits.append(-1); confidence.append(0.); continue
        s0=tone_score(t[mask],y[mask],f0); s1=tone_score(t[mask],y[mask],f1)
        bits.append(int(s1>s0)); confidence.append(abs(s1-s0))
    return np.asarray(bits), np.asarray(confidence)


def decode(t, y, config):
    symbol=float(config['symbol_seconds']); f0=float(config['f0_hz']); f1=float(config['f1_hz'])
    expected=np.asarray(config['bits'],dtype=int)
    fs=1/np.median(np.diff(t))
    if f1>=fs/2:
        raise ValueError('Camera sampling rate violates the Nyquist limit for f1.')
    last_start=t[-1]-len(PREAMBLE)*symbol
    if last_start<t[0]: raise ValueError('Recording is too short even for the preamble.')
    best=None
    # Synchronize using only the public preamble, never by matching payload truth.
    coarse=max(1/fs,symbol/12)
    for start in np.arange(t[0],last_start+1e-8,coarse):
        bits,conf=symbols_at(t,y,start,len(PREAMBLE),symbol,f0,f1)
        match=np.mean(bits==PREAMBLE)
        value=match+0.12*np.mean(conf)
        if best is None or value>best[0]: best=(value,start,match)
    coarse_start=best[1]
    for start in np.arange(max(t[0],coarse_start-coarse),min(last_start,coarse_start+coarse)+1e-8,1/(fs*4)):
        bits,conf=symbols_at(t,y,start,len(PREAMBLE),symbol,f0,f1)
        match=np.mean(bits==PREAMBLE); value=match+0.12*np.mean(conf)
        if value>best[0]: best=(value,start,match)
    received,confidence=symbols_at(t,y,best[1],len(expected),symbol,f0,f1)
    # A lost synchronization is a failed packet, not a fabricated perfect BER.
    sync=bool(best[2]>=12/13)
    correct=int(np.sum(received==expected)) if sync else 0
    payload_start=len(PREAMBLE)+8; payload_n=len(config['message'].encode('utf-8'))*8
    payload_rx=received[payload_start:payload_start+payload_n]
    payload_true=expected[payload_start:payload_start+payload_n]
    payload_correct=int(np.sum(payload_rx==payload_true)) if sync else 0
    message=None; crc_ok=False
    body=received[len(PREAMBLE):]
    if sync and len(body)%8==0 and np.all(body>=0):
        raw=bytes(int(''.join(map(str,body[i:i+8])),2) for i in range(0,len(body),8))
        if raw[0]==len(raw)-2:
            crc_ok=crc8(raw[:-1])==raw[-1]
            message=raw[1:-1].decode('utf-8',errors='replace')
    duration=len(expected)*symbol
    return {
        'status':'recorded_physical_trial','synchronized':sync,'preamble_correct_fraction':float(best[2]),
        'start_seconds':float(best[1]),'message_decoded':message,'crc_ok':crc_ok,
        'symbol_seconds':symbol,'f0_hz':f0,'f1_hz':f1,'delta_alpha':config['delta_alpha'],
        'distance_cm':config.get('distance_cm'),'nominal_raw_bps':1/symbol,
        'total_transmitted_bits':len(expected),'correct_bits':correct,
        'missing_or_erased_bits':int(np.sum(received<0)),
        'ber':1-correct/len(expected),'packet_duration_seconds':duration,
        'goodput_bps':correct/duration,'payload_goodput_bps':payload_correct/duration,
        'ber_definition':'All frame bits, including preamble/header/CRC; erasures count as errors. Sync failure counts all bits as lost.',
        'received_bits':received.tolist(),'confidence':confidence.tolist(),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path); p.add_argument('manifest',type=Path)
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','W','H'))
    p.add_argument('--out',type=Path,default=Path('results/communication/trial'))
    args=p.parse_args()
    config=json.loads(args.manifest.read_text(encoding='utf-8'))
    if config.get('status') != 'completed':
        raise ValueError('Use a completed transmitter manifest, not an aborted packet.')
    if config['bits'] != frame_bits(config['message']):
        raise ValueError('Manifest bits do not match its message/protocol.')
    if args.video.suffix.lower()=='.csv':
        trace=np.loadtxt(args.video,delimiter=',',skiprows=1,ndmin=2)
        if trace.shape[1]!=2 or len(trace)<3 or not np.all(np.isfinite(trace)) or not np.all(np.diff(trace[:,0])>0):
            raise ValueError('Receiver CSV must contain finite, increasing time_s and mean_gray columns.')
        t,values=trace[:,0],trace[:,1:2]
        fs=1/np.median(np.diff(t));timing='phone receiver CSV timestamps'
    else:
        if args.roi is None:p.error('Video input requires --roi X Y W H; receiver CSV does not.')
        t,values,fs,timing=read_video(args.video,args.roi)
    result=decode(t,values[:,0],config)
    result.update(video=str(args.video.resolve()),manifest=str(args.manifest.resolve()),roi=args.roi,sampling_hz=fs,timing=timing)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    with (args.out/'intensity.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['time_s','mean_gray']); w.writerows(zip(t,values[:,0]))
    fig,ax=plt.subplots(figsize=(11,3.5)); ax.plot(t,values[:,0],lw=.8)
    ax.axvline(result['start_seconds'],color='r',ls='--'); ax.set(xlabel='Recording time (s)',ylabel='ROI mean intensity',title=f"BFSK: BER={result['ber']:.3f}, goodput={result['goodput_bps']:.2f} bit/s")
    fig.tight_layout(); fig.savefig(args.out/'waveform.png',dpi=180); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
