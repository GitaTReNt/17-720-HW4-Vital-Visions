"""Review screen-camera recordings before choosing a decoding ROI."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image,ImageDraw

root=Path(__file__).resolve().parent
source=root/'data/heart_rate'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('videos',nargs='*')
parser.add_argument('--out',type=Path,default=root/'results/communication/input_review')
parser.add_argument('--times',type=float,nargs='+',default=[1,10,30,55])
args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
names=args.videos or ['alpha01_1bits_20cm.MOV','tx_d20_a0.5_t1.MOV']
times=args.times
sheet=Image.new('RGB',(len(times)*330,len(names)*370),'#e7edf2');draw=ImageDraw.Draw(sheet)
records=[]
for row,name in enumerate(names):
    path=source/name;cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():raise ValueError(f'Cannot open {path}')
    meta={'source_file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nominal_fps':fps,
          'frames':cap.get(cv2.CAP_PROP_FRAME_COUNT),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    meta['duration_seconds']=meta['frames']/fps
    records.append(meta)
    for col,t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
        if not ok:continue
        cv2.imwrite(str(out/f'{path.stem}_{t}s.jpg'),frame)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((310,300))
        x,y=col*330+10,row*370+60
        sheet.paste(im,(x,y));draw.text((x,row*370+10),name,fill='#172d3d')
        draw.text((x,row*370+30),f'{t} s; {meta["width"]} x {meta["height"]}',fill='#172d3d')
    cap.release()
sheet.save(out/'recording_review.jpg')
(out/'video_metadata.json').write_text(json.dumps(records,indent=2))
transmitters=[]
for name in names:
    log=(source/name).with_suffix('.json')
    if not log.exists():continue
    tx=json.loads(log.read_text())
    summary={k:tx.get(k) for k in ['status','message','symbol_seconds','f0_hz','f1_hz','delta_alpha','distance_cm','created_at','duration_seconds','display_refresh_hz']}
    summary['file']=str(log);summary['bit_count']=len(tx['bits']);summary['recorded_transitions']=len(tx.get('transitions',[]))
    transmitters.append(summary)
print(json.dumps({'videos':records,'transmitters':transmitters},indent=2))
