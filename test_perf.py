#!/usr/bin/env python3
"""性能检查：真实 send() + 模拟慢速长回复，量每帧重绘耗时，超预算就 FAIL。

    python3 test_perf.py && open /tmp/agent_perf.html

判据（60k 字符长回复）：中位帧 < 4ms、p95 < 8ms、超过 16.7ms 的帧 <= 2。
全量重绘的老实现是 20.6ms/帧、几乎每帧超预算——这里就是那条防回退线。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import PAGE  # noqa: E402

# 在 PAGE 的脚本之前插一段：包住 rAF，统计每个回调（=每次重绘）的耗时
HEAD = r"""<script>
window.__f=[];const _raf=window.requestAnimationFrame.bind(window);
window.requestAnimationFrame=cb=>_raf(t=>{const s=performance.now();cb(t);window.__f.push(performance.now()-s)});
</script>"""

BODY = r"""
<script>
const PAR='这是模型输出的中文字段落，带有 **粗体** 和 `code` 还有 [链接](https://x/y)。\n\n';
const big=n=>{let s='';while(s.length<n)s+=PAR;return s.slice(0,n)};
const run=async(total,len)=>{
  const doc=big(total),parts=[];
  for(let i=0;i<doc.length;i+=len)parts.push(doc.slice(i,i+len));
  const evs=parts.map(t=>'data: '+JSON.stringify({type:'text',delta:t})+'\n\n');
  evs.push('data: '+JSON.stringify({type:'done'})+'\n\n');
  const enc=new TextEncoder();
  const body=new ReadableStream({async start(c){
    for(const e of evs){c.enqueue(enc.encode(e));await new Promise(r=>setTimeout(r,8))}   // 8ms 一个 chunk，贴近真实节奏
    c.close();
  }});
  window.fetch=async()=>new Response(body,{status:200,headers:{'Content-Type':'text/event-stream'}});
  document.getElementById('log').textContent='';
  window.__f.length=0;
  const t0=performance.now();
  await send('长回复压测');
  const ms=performance.now()-t0;
  const f=[...window.__f].sort((a,b)=>a-b);
  const q=r=>f[Math.min(f.length-1,Math.floor(f.length*r))]||0;
  return {chars:doc.length,deltas:parts.length,ms,frames:f.length,
          med:q(.5),p95:q(.95),max:q(1),over:f.filter(x=>x>16.7).length,total:f.reduce((a,b)=>a+b,0)};
};
const out=[];let fail=0;
(async()=>{
  for(const [n,len] of [[20000,40],[60000,60]]){
    const r=await run(n,len);
    const ok=r.med<4&&r.p95<8&&r.over<=2;
    if(!ok)fail++;
    out.push((ok?'PASS ':'FAIL ')+r.chars+' 字符 / '+r.deltas+' delta / 墙钟 '+(r.ms/1000).toFixed(1)+'s');
    out.push('   rAF 帧数 '+r.frames+' | 中位 '+r.med.toFixed(2)+'ms | p95 '+r.p95.toFixed(2)+'ms | 最差 '+r.max.toFixed(2)+'ms | 超 16.7ms 的帧 '+r.over);
    out.push('   重绘占用 '+(r.total/1000).toFixed(2)+'s = 墙钟的 '+((r.total/r.ms)*100).toFixed(1)+'%');
  }
  document.title=fail?'PERF FAIL':'PERF PASS';
  document.body.innerHTML='<pre style="white-space:pre-wrap">'+out.join('\n')+'\n\n'+(fail?'PERF FAIL':'PERF PASS')+'</pre>';
})();
</script>
"""

page = PAGE.replace("<style>", HEAD + "<style>", 1).replace("</body></html>", BODY + "</body></html>")
open("/tmp/agent_perf.html", "w", encoding="utf-8").write(page)
print("wrote /tmp/agent_perf.html — 浏览器打开，标题 PERF PASS 才算通过（跑约 15 秒）")
