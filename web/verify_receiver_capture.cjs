const assert=require('node:assert/strict');
const {createCapture}=require('./receiver_capture.js');
function fixture(read=()=>120){
  let wall=0,tick,frame,cleared=false,cancelled=false;
  const values=[],errors=[];
  const video={readyState:4,videoWidth:720,currentTime:0,
    requestVideoFrameCallback(fn){frame=fn;return 1;},cancelVideoFrameCallback(){cancelled=true;}};
  const loop=createCapture(video,read,row=>values.push(row),message=>errors.push(message),
    {now:()=>wall,setInterval(fn){tick=fn;return 1;},clearInterval(){cleared=true;}});
  return {video,loop,values,errors,step(ms,withFrame=false){wall+=ms;if(withFrame)frame(wall,{mediaTime:0});tick();},
    stopped(){return cleared&&cancelled;}};
}
// A method existing without ever delivering callbacks must not stall sampling.
const stalled=fixture();stalled.loop.start();
for(let i=0;i<100;i++)stalled.step(34);
assert.ok(stalled.values.length>60);assert.equal(stalled.loop.stats().callbacks,0);
assert.equal(stalled.loop.stats().mode,'Timed capture (source FPS unknown)');
assert.ok(stalled.values.every((s,i)=>!i||s[0]>stalled.values[i-1][0]));
// Constant mediaTime in metadata must not suppress distinct callback deliveries.
const active=fixture();active.loop.start();
for(let i=0;i<100;i++)active.step(34,true);
assert.equal(active.values.length,100);assert.equal(active.loop.stats().callbacks,100);
active.loop.stop();const count=active.values.length;active.step(100,true);
assert.equal(active.values.length,count);assert.ok(active.stopped());
// A transient canvas read error is reported and subsequent callbacks recover.
let tries=0;const recover=fixture(()=>{if(++tries===1)throw Error('canvas unavailable');return 123;});recover.loop.start();
recover.step(34,true);recover.step(34,true);
assert.equal(recover.errors.length,1);assert.equal(recover.values.length,1);
console.log('PASS: stalled callback fallback, constant metadata clock, monotonic timestamps, stop, canvas error recovery.');
