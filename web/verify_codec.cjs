const assert=require('node:assert/strict');
const codec=require('./codec.js');
assert.equal(codec.crc8(new TextEncoder().encode('123456789')),0xf4);
for(const message of ['CMU','hello','你好']){
 const bits=codec.frame(message),T=1/3,start=1.73,samples=[];
 for(let i=0;i<Math.ceil((bits.length*T+4)*30);i++){
  const t=i/30,k=Math.floor((t-start)/T),bit=bits[k];
  samples.push([t,120+.1*t+(k>=0&&k<bits.length?12*Math.sin(2*Math.PI*((bit?4:2)/T)*(t-start-k*T)):0)]);
 }
 const r=codec.scan(samples,T);
 assert.ok(r,`${message}: packet not found`);assert.ok(r.crcOK,JSON.stringify(r));assert.equal(r.message,message);
}
assert.equal(codec.scan(Array.from({length:2100},(_,i)=>[i/30,120]),1),null);
console.log('JavaScript CRC + UTF-8 + synthetic BFSK round trips + no-signal rejection passed. Physical validation pending.');
