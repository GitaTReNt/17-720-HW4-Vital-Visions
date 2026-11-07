"""Synthetic checks only. These do not establish physical experiment performance."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys

import cv2
import numpy as np

from heart_rate import estimate, read_video
from communication import decode, frame_bits


def main():
    results=[]; rng=np.random.default_rng(42)
    fs=30.; t=np.arange(900)/fs
    x=100+3*np.sin(2*np.pi*1.2*t)+0.5*np.sin(2*np.pi*.1*t)+rng.normal(0,.08,len(t))
    _,_,_,peaks,hr=estimate(t,x,fs)
    assert abs(hr['fft_bpm']-72)<1
    assert abs(hr['time_domain_bpm']-72)<1
    assert 34<=len(peaks)<=37
    results.append({'check':'synthetic 72 bpm waveform','passed':True,'fft_bpm':hr['fft_bpm'],'peak_bpm':hr['time_domain_bpm']})
    out=Path('results/verification');out.mkdir(parents=True,exist_ok=True)
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='tmp') as temp:
        video=Path(temp)/'synthetic_72bpm.avi'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),fs,(96,64))
        assert writer.isOpened()
        for level in x: writer.write(np.full((64,96,3),round(level),np.uint8))
        writer.release()
        vt,v,vs,timing=read_video(video)
        _,_,_,_,vhr=estimate(vt,v[:,0],vs)
        assert abs(vhr['fft_bpm']-72)<1
        results.append({'check':'synthetic video decoding + grayscale + timing','passed':True,'fft_bpm':vhr['fft_bpm'],'timing':timing})
    for T in [1.,.5,1/3]:
        bits=frame_bits('CMU'); start=1.73
        t=np.arange(int((len(bits)*T+4)*30))/30
        t=t+rng.uniform(-.003,.003,len(t));t.sort()
        y=np.full(len(t),120.)
        for k,bit in enumerate(bits):
            mask=(t>=start+k*T)&(t<start+(k+1)*T)
            y[mask]+=12*np.sin(2*np.pi*((4 if bit else 2)/T)*(t[mask]-start-k*T))
        y+=.15*t+rng.normal(0,.5,len(t))
        config={'status':'completed','message':'CMU','bits':bits,'symbol_seconds':T,'f0_hz':2/T,'f1_hz':4/T,'delta_alpha':.1,'distance_cm':None}
        decoded=decode(t,y,config)
        assert decoded['crc_ok'], decoded
        assert decoded['ber']==0 and decoded['message_decoded']=='CMU',decoded
        results.append({'check':f'synthetic BFSK timing jitter + drift + noise at {1/T:g} bit/s','passed':True,'ber':decoded['ber'],'decoded':decoded['message_decoded']})
    with tempfile.TemporaryDirectory(dir='tmp') as temp:
        temp=Path(temp).resolve();trace=temp/'synthetic_receiver.csv';manifest=temp/'synthetic_tx.json'
        np.savetxt(trace,np.column_stack([t,y]),delimiter=',',header='time_s,mean_gray',comments='')
        manifest.write_text(json.dumps(config))
        command=[sys.executable,str(Path('communication.py').resolve()),str(trace),str(manifest),'--out',str(temp/'decoded')]
        run=subprocess.run(command,capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        cli=json.loads((temp/'decoded/result.json').read_text())
        assert cli['crc_ok'] and cli['ber']==0
        results.append({'check':'synthetic receiver CSV through the full decoding CLI','passed':True})
    t=np.arange(2000)/30
    lost=decode(t,np.full(len(t),120.),config)
    assert not lost['synchronized'] and lost['ber']==1 and lost['goodput_bps']==0
    results.append({'check':'constant input is a lost packet, not zero BER','passed':True})
    result={'status':'synthetic_algorithm_checks_only','checks':results,'physical_validation':False}
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
