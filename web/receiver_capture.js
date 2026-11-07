(function(root){
  'use strict';
  function createCapture(video,read,onSample,onError,env={}) {
    const now=env.now||(()=>performance.now()),schedule=env.setInterval||setInterval,clear=env.clearInterval||clearInterval;
    let active=false,epoch=0,lastCallback=0,lastRead=-Infinity,lastMedia=-1,lastFrames=-1,timer=null,handle=null;
    let mode='Waiting for video frames',count=0,callbacks=0;
    function take(kind){
      if(!active||video.readyState<2||!video.videoWidth)return;
      const wall=now();if(wall-lastRead<5)return;
      try{const value=read();if(!Number.isFinite(value))throw Error('Invalid camera intensity value');
        mode=kind;lastRead=wall;count++;onSample([(wall-epoch)/1000,value]);
      }catch(error){onError('Camera read failed: '+error.message);}
    }
    function frame(){
      if(!active)return;lastCallback=now();callbacks++;take('Video frame callback');
      if(active)handle=video.requestVideoFrameCallback(frame);
    }
    function tick(){
      if(!active||now()-lastCallback<650)return;
      const media=Number(video.currentTime);
      let frames=NaN;try{frames=video.getVideoPlaybackQuality?.().totalVideoFrames;}catch(_){}
      const changed=(Number.isFinite(media)&&media>lastMedia)||(Number.isFinite(frames)&&frames>lastFrames);
      // Timed canvas reads remain explicitly distinguished from camera frames.
      if(changed||now()-lastRead>=33.3){
        take(changed?'Compatible frame capture':'Timed capture (source FPS unknown)');
        if(Number.isFinite(media))lastMedia=media;if(Number.isFinite(frames))lastFrames=frames;
      }
    }
    return {
      start(){if(active)return;active=true;epoch=lastCallback=now();
        if(typeof video.requestVideoFrameCallback==='function')handle=video.requestVideoFrameCallback(frame);
        timer=schedule(tick,34);},
      stop(){active=false;if(timer!==null)clear(timer);if(handle!==null&&video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(handle);},
      stats(){return {count,callbacks,mode,elapsed:(now()-epoch)/1000};}
    };
  }
  if(typeof module!=='undefined')module.exports={createCapture};else root.ReceiverCapture={createCapture};
})(typeof window!=='undefined'?window:this);
