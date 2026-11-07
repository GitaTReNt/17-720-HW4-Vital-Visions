// Replay real recorded brightness traces through the browser decoder.
const fs=require('node:fs');
const path=require('node:path');
const codec=require('./web/codec.js');
const root=__dirname;
const summary=JSON.parse(fs.readFileSync(path.join(root,'results/communication/baseline_summary.json'),'utf8'));
let failed=0;
for(const row of summary.transmissions){
  const file=path.join(root,path.dirname(row.source.replace(/\\/g,'/')),'intensity.csv');
  const samples=fs.readFileSync(file,'utf8').trim().split(/\r?\n/).slice(1).map(l=>l.split(',').map(Number));
  const started=performance.now();
  const result=codec.scan(samples,1/row.nominal_raw_bps);
  const report={distance:row.distance_cm,alpha:row.delta_alpha,rate:row.nominal_raw_bps,
    milliseconds:Math.round(performance.now()-started),message:result?.message,crc:result?.crcOK,start:result?.start};
  if(!result?.crcOK||result.message!=='CMU')failed++;
  console.log(JSON.stringify(report));
}
if(failed)process.exitCode=1;
