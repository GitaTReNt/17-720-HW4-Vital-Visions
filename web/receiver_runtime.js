'use strict';
const $=id=>document.getElementById(id),video=$('video');
const ctx=$('canvas').getContext('2d',{willReadFrequently:true});
let samples=[],stream=null,capture=null,running=false,starting=false,session=0,after=0,lastScan=0,worker=null,busy=false;
let progress='Waiting for transmission',lastError='',heartbeat=null;
function makeWorker(){
  if(worker)worker.terminate();worker=new Worker('receiver_worker.js?v=english3');busy=false;
  worker.onmessage=event=>{
    if(event.data.session!==session)return;busy=false;
    if(event.data.error){lastError='Decoder error: '+event.data.error;return;}
    const result=event.data.result,p=event.data.progress;
    progress=p?.stage==='receiving'?`Preamble found; receiving packet (~${Math.ceil(p.remainingSeconds)} s remaining)`:'Searching for a preamble; click Send once on the computer';
    if(result){
      $('output').textContent=result.crcOK?`Received: ${result.message} | CRC PASS`:`Packet detected; CRC FAILED: ${result.message||result.error}`;
      $('output').className=result.crcOK?'success':'failure';after=result.end-.1*(+$('symbol').value);
      progress=result.crcOK?'Packet received successfully':'Packet check failed; save the CSV for analysis';
    }
  };
  worker.onerror=event=>{busy=false;lastError='Decoder worker error: '+event.message;};
}
function readBrightness(){
  const w=video.videoWidth,h=video.videoHeight;ctx.drawImage(video,w*.35,h*.35,w*.3,h*.3,0,0,64,64);
  const data=ctx.getImageData(0,0,64,64).data;let total=0;
  for(let i=0;i<data.length;i+=4)total+=.299*data[i]+.587*data[i+1]+.114*data[i+2];return total/4096;
}
function update(){
  if(!running)return;const stat=capture.stats(),T=+$('symbol').value;
  const tail=samples.slice(-120),span=tail.length>1?tail[tail.length-1][0]-tail[0][0]:0,hz=span>0?(tail.length-1)/span:0;
  $('status').textContent=`${samples.length} samples | ${stat.elapsed.toFixed(1)} s | ${hz.toFixed(1)} reads/s | ${stat.mode}`;
  if(lastError)$('detail').textContent=lastError;
  else if(!samples.length&&stat.elapsed>2)$('detail').textContent=`No samples yet: video state ${video.readyState}, size ${video.videoWidth}x${video.videoHeight}. Keep this page visible.`;
  else $('detail').textContent=progress+(samples.length&&hz<=8/T?'; low sampling rate: keep this page visible and improve lighting':'');
  if(samples.length>100&&!busy&&performance.now()-lastScan>1500){
    busy=true;lastScan=performance.now();const last=samples[samples.length-1][0],retained=samples.filter(s=>s[0]>=last-300*T-8);
    try{worker.postMessage({samples:retained,T,after,session});}catch(error){busy=false;lastError='Could not start decoding: '+error.message;}
  }
}
function stop(){
  running=false;capture?.stop();if(heartbeat)clearInterval(heartbeat);stream?.getTracks().forEach(track=>track.stop());
  worker?.terminate();worker=null;busy=false;session++;$('symbol').disabled=false;$('start').disabled=false;
  $('status').textContent=`Stopped; ${samples.length} samples retained`;
}
$('start').onclick=async()=>{
  if(running||starting)return;starting=true;$('start').disabled=true;
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw Error('Open this HTTPS page in Safari');
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'environment',frameRate:{ideal:30},width:{ideal:1280},height:{ideal:720}}});
    video.muted=true;video.playsInline=true;video.srcObject=stream;await video.play();
    samples=[];after=0;session++;lastScan=0;lastError='';progress='Waiting for transmission; confirm the sample count increases, then click Send once on the computer';
    $('output').textContent='No complete packet received yet';$('output').className='';$('symbol').disabled=true;
    makeWorker();running=true;
    capture=ReceiverCapture.createCapture(video,readBrightness,sample=>samples.push(sample),error=>{lastError=error;});
    capture.start();heartbeat=setInterval(update,250);update();
  }catch(error){stop();$('detail').textContent='Could not start reception: '+error.message;}
  finally{starting=false;if(!running)$('start').disabled=false;}
};
$('stop').onclick=stop;
$('save').onclick=()=>{
  if(!samples.length){$('detail').textContent='No samples captured. Check the status above.';return;}
  const blob=new Blob(['time_s,mean_gray\n'+samples.map(row=>row.join(',')).join('\n')],{type:'text/csv'});
  const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='receiver_intensity.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)lastError='This page is in the background; sampling may pause. Keep it visible.';else if(running)lastError='';});
window.addEventListener('error',event=>{$('detail').textContent='Page error: '+event.message;});
