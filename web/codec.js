(function (root) {
  'use strict';
  const preamble = [1,1,1,1,1,0,0,1,1,0,1,0,1];
  function crc8(bytes) {
    let crc=0;
    for (const byte of bytes) { crc ^= byte; for(let k=0;k<8;k++) crc=(crc&128)?((crc<<1)^7)&255:(crc<<1)&255; }
    return crc;
  }
  const byteBits = bytes => Array.from(bytes).flatMap(b => Array.from({length:8},(_,k)=>(b>>(7-k))&1));
  function frame(message) {
    const data=new TextEncoder().encode(message);
    if(data.length<1 || data.length>32) throw Error('Use 1-32 UTF-8 bytes.');
    const body=[data.length,...data];
    return [...preamble,...byteBits([...body,crc8(body)])];
  }
  function solve(a,b) {
    a=a.map((r,i)=>[...r,b[i]]);
    for(let j=0;j<b.length;j++) {
      let pivot=j; for(let i=j+1;i<b.length;i++) if(Math.abs(a[i][j])>Math.abs(a[pivot][j])) pivot=i;
      [a[j],a[pivot]]=[a[pivot],a[j]];
      if(Math.abs(a[j][j])<1e-10) return null;
      const d=a[j][j]; for(let k=j;k<=b.length;k++) a[j][k]/=d;
      for(let i=0;i<b.length;i++) if(i!==j) { const f=a[i][j]; for(let k=j;k<=b.length;k++) a[i][k]-=f*a[j][k]; }
    }
    return a.map(r=>r[b.length]);
  }
  function explained(samples,f) {
    if(samples.length<6) return 0;
    const begin=samples[0][0], mean=samples.reduce((a,s)=>a+s[1],0)/samples.length;
    const X=samples.map(s=>{const t=s[0]-begin; return [1,t,Math.sin(2*Math.PI*f*t),Math.cos(2*Math.PI*f*t)];});
    const matrix=Array.from({length:4},()=>Array(4).fill(0)), rhs=Array(4).fill(0);
    X.forEach((row,i)=>{for(let j=0;j<4;j++){rhs[j]+=row[j]*(samples[i][1]-mean);for(let k=0;k<4;k++)matrix[j][k]+=row[j]*row[k];}});
    const fit=solve(matrix,rhs); if(!fit) return 0;
    const total=samples.reduce((a,s)=>a+(s[1]-mean)**2,0);
    const residual=X.reduce((a,row,i)=>a+(samples[i][1]-mean-row.reduce((z,v,k)=>z+v*fit[k],0))**2,0);
    return Math.max(0,1-residual/(total+1e-12));
  }
  function symbols(samples,start,count,T,f0=2/T,f1=4/T) {
    const bits=[], confidence=[];
    for(let k=0;k<count;k++) {
      const a=start+(k+.08)*T,b=start+(k+.92)*T;
      const part=samples.filter(s=>s[0]>=a && s[0]<b);
      if(part.length<6) {bits.push(-1);confidence.push(0);continue;}
      const s0=explained(part,f0),s1=explained(part,f1);
      bits.push(Number(s1>s0));confidence.push(Math.abs(s1-s0));
    }
    return {bits,confidence};
  }
  function scan(samples,T,after=0,onProgress=()=>{}) {
    if(samples.length<100) return null;
    const last=samples[samples.length-1][0], max=last-(preamble.length+8)*T;
    let best=null;
    for(let t=Math.max(after,samples[0][0]);t<=max;t+=T/10) {
      const {bits,confidence}=symbols(samples,t,preamble.length,T);
      const match=bits.filter((b,k)=>b===preamble[k]).length;
      const score=match/preamble.length+.12*confidence.reduce((a,b)=>a+b,0)/preamble.length;
      if(!best || score>best.score)best={start:t,score,match};
    }
    if(!best || best.match<12)return null;
    const header=symbols(samples,best.start+preamble.length*T,8,T).bits;
    if(header.includes(-1))return null;
    const n=parseInt(header.join(''),2); if(n<1||n>32)return null;
    const count=preamble.length+8*(n+2), end=best.start+count*T;
    if(end>last){onProgress({stage:'receiving',start:best.start,end,remainingSeconds:end-last,payloadBytes:n});return null;}
    const decoded=symbols(samples,best.start,count,T), body=decoded.bits.slice(preamble.length);
    if(body.includes(-1))return {...best,end,crcOK:false,error:'Missing camera samples',...decoded};
    const bytes=[];for(let i=0;i<body.length;i+=8)bytes.push(parseInt(body.slice(i,i+8).join(''),2));
    return {...best,end,...decoded,crcOK:crc8(bytes.slice(0,-1))===bytes[bytes.length-1],message:new TextDecoder().decode(new Uint8Array(bytes.slice(1,-1)))};
  }
  const api={preamble,crc8,frame,symbols,scan};
  if(typeof module!=='undefined')module.exports=api; else root.BFSK=api;
})(typeof window!=='undefined'?window:this);
