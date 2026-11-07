'use strict';
importScripts('codec.js?v=receiver2');
self.onmessage=event=>{
  const {samples,T,after,session}=event.data;let progress={stage:'search'};
  try{const result=BFSK.scan(samples,T,after,value=>{progress=value;});self.postMessage({session,result,progress});}
  catch(error){self.postMessage({session,error:error.message});}
};
