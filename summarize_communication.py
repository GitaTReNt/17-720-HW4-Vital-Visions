"""Combine measured trial results. Empty conditions stay missing, never zero."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from communication import frame_bits

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('results/communication'))
p.add_argument('--unlogged-expected-message',help='Expected message established independently before decoding; never infer it from the received payload.')
args=p.parse_args();rows=[]
for path in args.root.glob('*/result.json'):
    r=json.loads(path.read_text())
    if r.get('status')=='recorded_physical_trial':
        reference_type='original_transmitter_log'
    elif r.get('status')=='recorded_video_without_transmitter_log' and args.unlogged_expected_message:
        expected=frame_bits(args.unlogged_expected_message)
        received=r['received_bits']
        correct=sum(a==b for a,b in zip(expected,received)) if r['synchronized'] else 0
        duration=len(expected)*r['symbol_seconds']
        payload_start=21;payload_length=8*len(args.unlogged_expected_message.encode('utf-8'))
        payload_correct=sum(a==b for a,b in zip(expected[payload_start:payload_start+payload_length],received[payload_start:payload_start+payload_length])) if r['synchronized'] else 0
        reference_type='prior_fixed_protocol_without_original_tx_log'
        comparison={'reference_type':reference_type,'expected_message':args.unlogged_expected_message,
                    'reference_source':'CMU was specified in the experimental setup before the first recording; no original transmitter timing record was saved.',
                    'expected_bits':expected,'correct_bits':correct,'total_transmitted_bits':len(expected),
                    'ber':1-correct/len(expected),'goodput_bps':correct/duration,'payload_goodput_bps':payload_correct/duration,
                    'caveat':'This comparison is conditional on the predeclared CMU transmission configuration. It is not an original transmitter log and does not recover refresh timing.'}
        (path.parent/'protocol_reference_comparison.json').write_text(json.dumps(comparison,indent=2))
        r={**r,**comparison}
    else:continue
    rows.append({k:r.get(k) for k in ['distance_cm','delta_alpha','nominal_raw_bps','ber','goodput_bps','payload_goodput_bps','crc_ok','synchronized','message_decoded','correct_bits','total_transmitted_bits']}|{'reference_type':reference_type,'source':str(path)})
if not rows: raise SystemExit('No physical trials found. No plots or zero-valued results were fabricated.')
rows.sort(key=lambda row:(row['nominal_raw_bps'],row['distance_cm'],row['delta_alpha']))
with (args.root/'measured_trials.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for metric,label in [('ber','Bit error rate'),('goodput_bps','Correctly received frame bits / second')]:
    fig,ax=plt.subplots(figsize=(8,5))
    for distance in sorted({r['distance_cm'] for r in rows if r['distance_cm'] is not None}):
        for rate in sorted({r['nominal_raw_bps'] for r in rows}):
            group=[r for r in rows if r['distance_cm']==distance and r['nominal_raw_bps']==rate]
            group.sort(key=lambda r:r['delta_alpha'])
            if group: ax.plot([r['delta_alpha'] for r in group],[r[metric] for r in group],'o',label=f'{distance} cm, {rate:g} raw bit/s')
    ax.set(xlabel='Alpha difference',ylabel=label,title='Measured screen-camera trials (individual repeats)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(args.root/f'{metric}_vs_alpha.png',dpi=170);plt.close(fig)
baseline=[r for r in rows if r['nominal_raw_bps']==1]
live_path=args.root/'real_time_reception.json'
live=json.loads(live_path.read_text()) if live_path.exists() else {}
live_success=live.get('status') in ['user_confirmed_live_reception_success','recorded_live_reception_success']
summary={'status':'baseline_recordings_analyzed','baseline_conditions':len(baseline),
         'required_baseline_conditions':8,'baseline_matrix_complete':{(r['distance_cm'],r['delta_alpha']) for r in baseline}=={(d,a) for d in [20,40,80,120] for a in [.1,.5]},
         'transmissions':rows,'observed_frame_bit_errors':sum(r['total_transmitted_bits']-r['correct_bits'] for r in rows),
         'total_frame_bits_compared':sum(r['total_transmitted_bits'] for r in rows),
         'faster_rate_trials':sum(r['nominal_raw_bps']>1 for r in rows),'phone_real_time_reception_demonstrated':live_success,
         'phone_real_time_reception_evidence':live.get('evidence_type'),
         'limitations':['One packet per baseline condition; zero observed errors is not a guarantee of zero channel error probability.',
                        '20 cm / alpha 0.1 uses the predeclared CMU protocol reference rather than a saved original transmitter log.',
                        'Display layout and selected ROI differ across distance recordings; the 80/120 cm transmitter is windowed, so distance effects are confounded with display geometry.',
                        'Offline BER measurements are separate from the live CMU / CRC-success demo. Its screen recording and extracted frames are archived; a live intensity CSV and demo transmitter log are unavailable.' if live.get('status')=='recorded_live_reception_success' else 'Offline BER measurements are separate from a participant-confirmed live CMU / CRC-success test; its screenshot and live CSV are not yet archived.' if live_success else 'The phone recordings were decoded offline. Phone real-time reception has not been demonstrated.',
                        'Higher-rate trials cover only 120 cm / alpha 0.1, with one packet at each of 2 and 3 raw bit/s.']}
(args.root/'baseline_summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,4,figsize=(12,5.5),sharex=True,sharey='row')
for col,distance in enumerate([20,40,80,120]):
    group=[r for r in baseline if r['distance_cm']==distance]
    group.sort(key=lambda row:row['delta_alpha'])
    for row,(key,label) in enumerate([('ber','Observed BER'),('goodput_bps','Correct frame bit/s')]):
        ax=axes[row,col];ax.plot([r['delta_alpha'] for r in group],[r[key] for r in group],'o-',color='#146a71')
        ax.set(xticks=[.1,.5],xlim=(0,.6),title=f'{distance} cm' if row==0 else None,xlabel='Alpha difference' if row else None)
        if col==0:ax.set_ylabel(label)
        ax.set_ylim((-.005,.05) if row==0 else (0,1.1));ax.grid(alpha=.2)
fig.suptitle('Eight baseline conditions at 1 raw bit/s (one packet each)')
fig.tight_layout();fig.savefig(args.root/'baseline_matrix.png',dpi=180);plt.close(fig)
print(f'Summarized {len(rows)} physical trials.')
