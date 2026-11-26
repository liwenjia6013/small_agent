#!/usr/bin/env python3
"""Front-end probe: 从 agent.py 里真实的 PAGE 生成一个自检页，浏览器打开看 PASS/FAIL。

    python3 test_frontend_probe.py      # 生成 /tmp/agent_probe.html
    open /tmp/agent_probe.html          # 全 PASS 才对

为什么不在 selftest 里做：清洗函数依赖 DOMParser，标准库里没有 DOM。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import PAGE  # noqa: E402

PROBE = r"""
<script>
const out=[];
const t=(n,f)=>{try{const v=f();out.push((v?'PASS ':'FAIL ')+n+(v?'':' -> '+JSON.stringify(String(v))))}catch(e){out.push('FAIL '+n+' threw '+e)}};
t('marked loaded',()=>typeof marked.parse==='function');
t('md bold',()=>html('**b**').includes('<strong>b</strong>'));
t('md code fence',()=>html('```\nx=1\n```').includes('<code'));
t('md list',()=>html('- a\n- b').includes('<li>'));
t('no raw html passthrough',()=>!/onerror/i.test(html('hi <img src=x onerror=alert(1)>')));
t('style stripped',()=>!/style=/i.test(html('<div style="background:url(//evil/x)">a</div>')));
t('md image still works',()=>/<img[^>]+src="https:\/\/x\/y.png"/.test(html('![a](https://x/y.png)')));
t('no script',()=>!html('<script>alert(1)<\/script>').includes('<script'));
t('no js href',()=>!/href\s*=\s*"javascript:/i.test(html('[x](javascript:alert(1))')));
t('lt stays text',()=>html('a < b').includes('&lt;'));
t('data: img blocked',()=>!/<img[^>]+src="data:/i.test(html('![a](data:text/html;base64,xx)')));
t('iframe unwrapped',()=>!html('<iframe src=//evil></iframe>ok').includes('iframe'));
t('svg onload stripped',()=>!/onload/i.test(html('<svg onload=alert(1)></svg>')));
toolCard({id:'c1',name:'<img src=x onerror=alert(1)>',args:{path:'<b>pwn</b>'}});
t('toolCard no markup',()=>{const d=document.querySelector('.tool');return !!d&&!d.querySelector('img')&&d.textContent.includes('<img src=x')});
t('toolCard name is text',()=>document.querySelector('.tool b').textContent.startsWith('<img'));
// 轮次分组：带工具的一轮留折叠壳，纯文字一轮拆壳平铺
document.getElementById('log').textContent='';
const g=el('details','round');g.open=true;const s=el('summary');s.textContent='第 1 轮 · bash';g.appendChild(s);
const gb=el('div','round-body');g.appendChild(gb);document.getElementById('log').appendChild(g);
bubble('assistant',gb).textContent='先说话';toolCard({id:'c2',name:'bash',args:{command:'ls'}},gb);
const plainG=el('details','round');plainG.open=true;const pbody=el('div','round-body');plainG.appendChild(pbody);
document.getElementById('log').appendChild(plainG);bubble('assistant',pbody).textContent='最终回答';
t('group holds text+tool',()=>{const r=document.querySelector('.round');return !!r&&!!r.querySelector('.bubble')&&!!r.querySelector('.tool')});
t('tool inside round-body',()=>document.querySelector('.tool').parentElement.className==='round-body');
t('collapse works',()=>{const r=document.querySelector('.round');r.open=false;const h=r.offsetHeight;r.open=true;return h<r.offsetHeight});
t('unwrap plain round',()=>{const f=document.createDocumentFragment();while(pbody.firstChild)f.appendChild(pbody.firstChild);
  plainG.replaceWith(f);return !document.querySelectorAll('.round')[1]&&document.querySelectorAll('#log>.msg').length>=1});
document.title='PROBE';
document.body.innerHTML='<pre id="r">'+out.join('\n')+'</pre>';
</script>
"""

out = "/tmp/agent_probe.html"
assert "</body></html>" in PAGE
open(out, "w", encoding="utf-8").write(PAGE.replace("</body></html>", PROBE + "</body></html>"))
print("wrote %s — 用浏览器打开，全部 PASS 才算通过" % out)

# 第二个页：假 SSE 流喂 send()，把每轮一个 <details> 分组的 DOM 结构 dump 出来
SIM = r"""
<script>
const EV=[
 {type:'text',delta:'我先跑第一条。'},
 {type:'tool',id:'t1',name:'bash',args:{command:'echo a'}},
 {type:'tool_result',id:'t1',output:'exit=0\na'},
 {type:'usage',usage:{prompt:100,completion:40,total:140}},
 {type:'text',delta:'第一条跑完，接着第二条。'},
 {type:'tool',id:'t2',name:'bash',args:{command:'echo b'}},
 {type:'tool_result',id:'t2',output:'exit=0\nb'},
 {type:'usage',usage:{prompt:200,completion:60,total:260}},
 {type:'text',delta:'总结：a 和 b 都出来了。'},
 {type:'usage',usage:{prompt:300,completion:20,total:320}},
 {type:'done'},
];
window.fetch=async(url)=>{
  if(String(url).includes('/api/config'))return new Response('{"max_rounds":20}',{status:200,headers:{'Content-Type':'application/json'}});
  return new Response(EV.map(e=>'data: '+JSON.stringify(e)+'\n\n').join(''),{status:200,headers:{'Content-Type':'text/event-stream'}});
};
const dump=(n,d,out)=>{
  const pad='  '.repeat(d);
  for(const c of n.children){
    if(c.classList.contains('msg')) out.push(pad+'MSG.'+c.className.replace('msg','').trim()+' 「'+(c.querySelector('.bubble').textContent||'').slice(0,24)+'」');
    else if(c.classList.contains('round')) out.push(pad+'DETAILS.round open='+c.open+' summary="'+c.querySelector('summary').textContent+'"');
    else if(c.classList.contains('tool')) out.push(pad+'TOOL '+c.querySelector('b').textContent);
    else out.push(pad+c.tagName+'.'+c.className);
    dump(c,d+1,out);
  }
  return out;
};
setTimeout(async()=>{
  const out=[];
  await send('三步测试');
  await new Promise(r=>setTimeout(r,120));
  dump(document.getElementById('log'),0,out);
  const rd=[...document.querySelectorAll('.round')];
  const first=rd[0]&&rd[0].querySelector('summary').textContent;
  out.push('CHK 轮数='+rd.length+(rd.length===2?' PASS':' FAIL'));
  out.push('CHK 收尾后工具组全部收起='+(rd.every(r=>!r.open)?'PASS':'FAIL'));
  out.push('CHK 标题带耗时='+(/\d+\.\ds/.test(first)&&/bash/.test(first)?'PASS':'FAIL')+'  「'+first+'」');
  out.push('CHK 标题带 token='+(/140 tok/.test(first)?'PASS':'FAIL'));
  const fb=document.getElementById('foldBtn');
  out.push('CHK 折叠按钮可见='+(fb&&!fb.classList.contains('hidden')?([...document.querySelectorAll('.round')].some(r=>r.open)?'PASS 全部收起':'PASS 全部展开'):'FAIL'));
  fb.click();
  out.push('CHK 一键展开='+(document.querySelectorAll('.round[open]').length===2?'PASS':'FAIL')+' 按钮="'+fb.textContent+'"');
  fb.click();
  out.push('CHK 一键收起='+(document.querySelectorAll('.round[open]').length===0?'PASS':'FAIL')+' 按钮="'+fb.textContent+'"');

  // 增量重绘不丢字/不重复：造一段带代码围栏、多空行块的流，逐 delta 喂进去
  const DOC='# 标题\n\n第一段 **粗体**。\n\n第二段 `code`。\n\n```js\nconst a=1;\nconst b=2;\n```\n\n- 条目一\n- 条目二\n\n最后一段。\n';
  const parts=[];for(let i=0;i<DOC.length;i+=7)parts.push(DOC.slice(i,i+7));
  window.fetch=async()=>new Response(parts.map(t=>'data: '+JSON.stringify({type:'text',delta:t})+'\n\n').join('')
    +'data: '+JSON.stringify({type:'done'})+'\n\n',{status:200,headers:{'Content-Type':'text/event-stream'}});
  document.getElementById('log').textContent='';
  await send('长文档');
  await new Promise(r=>setTimeout(r,150));
  const got=document.querySelector('.msg.assistant .bubble').textContent.replace(/\s+/g,'');
  // 可见文本：围栏语言标记(js)、列表符号、#/*/` 都不会出现在渲染结果里
  const want='标题第一段粗体。第二段code。consta=1;constb=2;条目一条目二最后一段。';
  out.push('CHK 增量重绘无丢字/重复='+(got===want?'PASS':'FAIL'));
  if(got!==want)out.push('   got  '+(got.length>120?got.slice(0,60)+' … '+got.slice(-40):got)+'\n   want '+(want.length>120?want.slice(0,60)+' … '+want.slice(-40):want));
  out.push('CHK 围栏仍是单个 pre='+(document.querySelectorAll('.msg.assistant pre').length===1?'PASS':'FAIL'));
  out.push('CHK 标题是 h1='+(document.querySelectorAll('.msg.assistant h1').length===1?'PASS':'FAIL'));
  out.push('CHK 列表项数=2 实际='+document.querySelectorAll('.msg.assistant li').length);

  // 主动停止：慢速流 + 点「停止」→ 部分文本保留、按钮复位、能继续发
  window.fetch=async(url,opt)=>{
    if(String(url).includes('/api/stop'))return new Response('{"ok":true}',{status:200,headers:{'Content-Type':'application/json'}});
    const enc=new TextEncoder();
    const body=new ReadableStream({start(c){
      c.enqueue(enc.encode('data: '+JSON.stringify({type:'text',delta:'半截话'})+'\n\n'));
      opt.signal.addEventListener('abort',()=>c.error(new DOMException('aborted','AbortError')));
    }});
    return new Response(body,{status:200,headers:{'Content-Type':'text/event-stream'}});
  };
  document.getElementById('log').textContent='';
  const sb=document.getElementById('stop');
  const p=send('停止测试');
  await new Promise(r=>setTimeout(r,150));
  out.push('CHK 回复中显示停止='+(!sb.classList.contains('hidden')?'PASS':'FAIL'));
  sb.click();
  await p;
  await new Promise(r=>setTimeout(r,80));
  out.push('CHK 停止后按钮隐藏='+(sb.classList.contains('hidden')?'PASS':'FAIL'));
  out.push('CHK 停止后有提示='+(/已停止/.test(document.getElementById('log').textContent)?'PASS':'FAIL'));
  out.push('CHK 部分输出保留='+((document.querySelector('.msg.assistant .bubble')||{}).textContent?.includes('半截话')?'PASS':'FAIL'));
  out.push('CHK 停止后可再输入='+(!document.getElementById('b').disabled?'PASS':'FAIL'));
  out.push('CHK 停止不算报错='+(!document.querySelector('.err')?'PASS':'FAIL'));
  document.title='SIM';
  document.body.innerHTML='<pre id="r">'+out.join('\n')+'</pre>';
},100);
</script>
"""
sim = "/tmp/agent_sim.html"
open(sim, "w", encoding="utf-8").write(PAGE.replace("</body></html>", SIM + "</body></html>"))
print("wrote %s — 轮次分组结构（应两轮 DETAILS + 末尾平铺 MSG）" % sim)
