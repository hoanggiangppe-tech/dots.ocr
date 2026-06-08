#!/usr/bin/env python3
"""
dots.ocr Kaggle Manager
Quản lý nhiều tài khoản Kaggle để chạy vLLM server 24/7.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json, os, re, sys, base64, threading, time, shutil, tempfile
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path.home() / '.dots_ocr_manager'
CONFIG_FILE = APP_DIR / 'config.json'
URL_FILE = APP_DIR / 'current_url.txt'
APP_DIR.mkdir(parents=True, exist_ok=True)

MAX_HOURS = 28          # switch trước khi đạt 30h/tuần
POLL_INTERVAL = 30      # giây giữa các lần check relay
KAGGLE_API = 'https://www.kaggle.com/api/v1'
KERNEL_SLUG = 'dots-ocr-server'

# ─── Kernel script chạy trên Kaggle ───────────────────────────────────────────
KERNEL_SCRIPT = r'''
import subprocess, time, os, json
from datetime import datetime
import requests as _req

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

NGROK_TOKEN = "{NGROK_TOKEN}"
NPOINT_ID   = "{NPOINT_ID}"
BRANCH      = "claude/setup-local-repo-4LmIb"
REPO_URL    = "https://github.com/hoanggiangppe-tech/dots.ocr.git"

def relay(url, status):
    if not NPOINT_ID: return
    try: _req.post(f"https://api.npoint.io/{NPOINT_ID}",
                   json={"url": url, "status": status, "ts": time.time()}, timeout=8)
    except: pass

relay("", "starting")

# ── Cài packages ─────────────────────────────────────────────────────────────
log("Installing packages...")
subprocess.run(["pip","install","vllm>=0.11.0,<0.15.0","pyngrok","xformers","-q"])
import torch
cv = torch.version.cuda.replace(".","")
tv = ".".join(torch.__version__.split(".")[:2])
subprocess.run(["pip","install","flashinfer-python","-i",
                f"https://flashinfer.ai/whl/cu{cv}/torch{tv}/","-q"])
log("Packages done")

# ── Clone repo + model ────────────────────────────────────────────────────────
if not os.path.exists('/kaggle/working/dots.ocr'):
    subprocess.run(["git","clone",REPO_URL,"/kaggle/working/dots.ocr"])
os.chdir('/kaggle/working/dots.ocr')
subprocess.run(["git","checkout",BRANCH])
subprocess.run(["git","pull","origin",BRANCH])
if not os.path.exists('./weights/DotsMOCR'):
    log("Downloading model (~4GB)...")
    subprocess.run(["python3","tools/download_model.py"])
log("Repo+model ready")

# ── Patch model files ─────────────────────────────────────────────────────────
def patch_file(fp):
    with open(fp) as f: lines = f.readlines()
    out, mod, i = [], False, 0
    while i < len(lines):
        l = lines[i]; s = l.lstrip(); ind = l[:len(l)-len(s)]
        if any(f'Auto{t}.register' in l for t in ['Config','Model','Processor','Tokenizer']):
            if i == 0 or 'try:' not in lines[i-1]:
                out += [f'{ind}try:\n',f'{ind}    {s}',
                        f'{ind}except (ValueError,AssertionError):\n',f'{ind}    pass\n']
                mod=True; i+=1; continue
        out.append(l); i+=1
    if mod:
        with open(fp,'w') as f: f.writelines(out)

FLASH_FALLBACK = """try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    import torch, math
    def flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,
                               dropout_p=0.0,softmax_scale=None,causal=False,**kw):
        if softmax_scale is None: softmax_scale=1.0/math.sqrt(q.shape[-1])
        out=[]
        for b in range(len(cu_seqlens_q)-1):
            qs,qe=cu_seqlens_q[b].item(),cu_seqlens_q[b+1].item()
            ks,ke=cu_seqlens_k[b].item(),cu_seqlens_k[b+1].item()
            o=torch.nn.functional.scaled_dot_product_attention(
                q[qs:qe].transpose(0,1).unsqueeze(0),k[ks:ke].transpose(0,1).unsqueeze(0),
                v[ks:ke].transpose(0,1).unsqueeze(0),scale=softmax_scale,is_causal=causal)
            out.append(o.squeeze(0).transpose(0,1))
        return torch.cat(out,dim=0)"""

for root,_,files in os.walk('./weights/DotsMOCR'):
    for fn in files:
        if fn.endswith('.py'):
            fp=os.path.join(root,fn); patch_file(fp)
            with open(fp) as f: c=f.read()
            if 'from flash_attn import flash_attn_varlen_func' in c and 'except ImportError' not in c:
                with open(fp,'w') as f:
                    f.write(c.replace('from flash_attn import flash_attn_varlen_func',FLASH_FALLBACK.strip()))

import shutil
cache=os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(cache): shutil.rmtree(cache)
log("Model patched")

# ── libcuda stub ──────────────────────────────────────────────────────────────
with open('/tmp/cuda_stub.c','w') as f: f.write('void _cuda_stub_(void){}\n')
if subprocess.run(['gcc','-shared','-fPIC','-o','/tmp/libcuda.so','/tmp/cuda_stub.c'],
                  capture_output=True).returncode == 0:
    subprocess.run(['rm','-rf','/root/.cache/flashinfer'],capture_output=True)
    os.environ['LIBRARY_PATH']='/tmp'

# ── Patch FlashInfer ──────────────────────────────────────────────────────────
fi='/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py'
_M='# _SDPA_FALLBACK_PATCHED_'
try:
    with open(fi) as f: c=f.read()
    if _M not in c:
        OLD=('                    prefill_wrapper.run(\n'
             '                        prefill_query,\n'
             '                        kv_cache_permute,\n'
             '                        k_scale=layer._k_scale_float,\n'
             '                        v_scale=layer._v_scale_float,\n'
             '                        out=out_prefill,\n'
             '                        kv_cache_sf=kv_cache_sf,\n'
             '                    )')
        NEW=('                    '+_M+'\n'
             '                    try:\n'
             '                        prefill_wrapper.run(\n'
             '                            prefill_query,\n'
             '                            kv_cache_permute,\n'
             '                            k_scale=layer._k_scale_float,\n'
             '                            v_scale=layer._v_scale_float,\n'
             '                            out=out_prefill,\n'
             '                            kv_cache_sf=kv_cache_sf,\n'
             '                        )\n'
             '                    except Exception:\n'
             '                        import torch.nn.functional as _F\n'
             '                        _q=prefill_query.float()\n'
             '                        _k=key[num_decode_tokens:].float()\n'
             '                        _v=value[num_decode_tokens:].float()\n'
             '                        _qt=_q.transpose(0,1).unsqueeze(0)\n'
             '                        _kt=_k.transpose(0,1).unsqueeze(0)\n'
             '                        _vt=_v.transpose(0,1).unsqueeze(0)\n'
             '                        _nh,_nkvh=_qt.shape[1],_kt.shape[1]\n'
             '                        if _nh!=_nkvh:\n'
             '                            _kt=_kt.repeat_interleave(_nh//_nkvh,dim=1)\n'
             '                            _vt=_vt.repeat_interleave(_nh//_nkvh,dim=1)\n'
             '                        _o=_F.scaled_dot_product_attention(_qt,_kt,_vt,is_causal=True,scale=self.scale)\n'
             '                        out_prefill.copy_(_o.squeeze(0).transpose(0,1).to(out_prefill.dtype))')
        if OLD in c:
            with open(fi,'w') as f: f.write(c.replace(OLD,NEW))
            log("FlashInfer patched")
except Exception as e: log(f"FlashInfer patch skip: {e}")

# ── Tạo ngrok tunnel ──────────────────────────────────────────────────────────
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN
ngrok.kill(); time.sleep(2)
tunnel = ngrok.connect(8000, bind_tls=True)
public_url = tunnel.public_url
log(f"Tunnel: {public_url}")
relay(public_url, "tunnel_up")

# ── Start vLLM ────────────────────────────────────────────────────────────────
env=os.environ.copy()
env.update({'VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS',
            'VLLM_USE_FLASHINFER_SAMPLER':'0','LIBRARY_PATH':'/tmp'})
lf=open('/tmp/vllm.log','w')
proc=subprocess.Popen([
    'vllm','serve','./weights/DotsMOCR',
    '--tensor-parallel-size','1','--gpu-memory-utilization','0.95',
    '--max-model-len','32768','--dtype','half',
    '--chat-template-content-format','string','--served-model-name','model',
    '--trust-remote-code','--enforce-eager','--port','8000',
],stdout=lf,stderr=lf,env=env,preexec_fn=os.setsid)

log("Waiting for vLLM...")
for i in range(120):
    time.sleep(5)
    if proc.poll() is not None:
        log(f"vLLM crashed: {proc.poll()}")
        with open('/tmp/vllm.log') as f: print(f.read()[-1500:],flush=True)
        relay(public_url,"error"); sys.exit(1)
    try:
        if _req.get('http://localhost:8000/v1/models',timeout=3).status_code==200:
            log(f"vLLM ready after {(i+1)*5}s!"); relay(public_url,"ready"); break
    except: pass
else:
    log("vLLM timeout"); relay(public_url,"error")

# ── Keep-alive ────────────────────────────────────────────────────────────────
count=0
while True:
    time.sleep(60); count+=1
    try: ok=_req.get('http://localhost:8000/v1/models',timeout=5).status_code==200
    except: ok=False
    if count%5==0:
        log(f"[{count}m] {'OK' if ok else 'FAIL'} {public_url}")
        relay(public_url,"ready" if ok else "error")
'''

# ─── Config ───────────────────────────────────────────────────────────────────
class Config:
    DEFAULTS = {
        'ngrok_token': '',
        'npoint_id': '',
        'accounts': [],
    }

    def __init__(self):
        self.data = dict(self.DEFAULTS)
        self.load()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def __getitem__(self, k): return self.data[k]
    def __setitem__(self, k, v): self.data[k] = v


# ─── Kaggle API client ────────────────────────────────────────────────────────
class KaggleClient:
    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
        creds = base64.b64encode(f'{username}:{api_key}'.encode()).decode()
        self.headers = {'Authorization': f'Basic {creds}'}

    def _get(self, path, **kw):
        return requests.get(f'{KAGGLE_API}{path}', headers=self.headers, timeout=30, **kw)

    def _post(self, path, **kw):
        return requests.post(f'{KAGGLE_API}{path}', headers=self.headers, timeout=60, **kw)

    def push_kernel(self, script_content: str) -> dict:
        meta = {
            'id': f'{self.username}/{KERNEL_SLUG}',
            'title': 'dots-ocr-server',
            'code_file': 'script.py',
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': True,
            'enable_internet': True,
            'dataset_data_sources': [],
            'kernel_data_sources': [],
        }
        r = self._post('/kernels/push', files={
            'blob': (None, json.dumps({'metadata': meta, 'blob': script_content})),
        })
        return r.json() if r.ok else {'error': r.text}

    def kernel_status(self) -> str:
        r = self._get(f'/kernels/{self.username}/{KERNEL_SLUG}/status')
        if r.ok:
            return r.json().get('status', 'unknown')
        return 'unknown'

    def test_auth(self) -> bool:
        try:
            r = self._get('/competitions/list')
            return r.status_code in (200, 403)
        except Exception:
            return False


# ─── Account tracker ──────────────────────────────────────────────────────────
class AccountTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()

    def accounts(self):
        return self.cfg['accounts']

    def active_account(self):
        for acc in self.accounts():
            if acc.get('active'):
                return acc
        return None

    def reset_week_if_needed(self):
        now = datetime.now()
        monday = now - timedelta(days=now.weekday())
        week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        changed = False
        for acc in self.accounts():
            ws = acc.get('week_start')
            if ws:
                acc_week = datetime.fromisoformat(ws)
                if acc_week < week_start:
                    acc['hours_used'] = 0.0
                    acc['week_start'] = week_start.isoformat()
                    changed = True
            else:
                acc['hours_used'] = 0.0
                acc['week_start'] = week_start.isoformat()
                changed = True
        if changed:
            self.cfg.save()

    def best_account(self):
        self.reset_week_if_needed()
        available = [a for a in self.accounts()
                     if a.get('hours_used', 0) < MAX_HOURS]
        if not available:
            return None
        return min(available, key=lambda a: a.get('hours_used', 0))

    def add_hours(self, account_name: str, hours: float):
        with self._lock:
            for acc in self.accounts():
                if acc['name'] == account_name:
                    acc['hours_used'] = acc.get('hours_used', 0) + hours
                    self.cfg.save()
                    break

    def set_active(self, account_name: str):
        with self._lock:
            for acc in self.accounts():
                acc['active'] = (acc['name'] == account_name)
            self.cfg.save()

    def hours_remaining(self, account_name: str) -> float:
        for acc in self.accounts():
            if acc['name'] == account_name:
                return max(0.0, MAX_HOURS - acc.get('hours_used', 0))
        return 0.0


# ─── GUI ──────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('dots.ocr Kaggle Manager')
        self.geometry('820x680')
        self.resizable(True, True)
        self.configure(bg='#1e1e2e')

        self.cfg = Config()
        self.tracker = AccountTracker(self.cfg)
        self._running = False
        self._session_start: datetime | None = None
        self._current_account: dict | None = None
        self._current_url = ''
        self._npoint_id = self.cfg['npoint_id']
        self._stop_event = threading.Event()

        self._build_ui()
        self._refresh_table()
        self._restore_url()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        DARK, DARKER, ACCENT = '#1e1e2e', '#181825', '#89b4fa'
        FG, FG2 = '#cdd6f4', '#a6adc8'
        GREEN, YELLOW, RED = '#a6e3a1', '#f9e2af', '#f38ba8'

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Treeview', background=DARKER, fieldbackground=DARKER,
                        foreground=FG, rowheight=26)
        style.configure('Treeview.Heading', background='#313244', foreground=ACCENT,
                        font=('Segoe UI', 9, 'bold'))
        style.map('Treeview', background=[('selected', '#45475a')])
        style.configure('TProgressbar', troughcolor=DARKER, background=ACCENT)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg='#313244', pady=8)
        hdr.pack(fill='x')
        tk.Label(hdr, text='  🔍 dots.ocr Kaggle Manager', bg='#313244',
                 fg=ACCENT, font=('Segoe UI', 14, 'bold')).pack(side='left')
        self._status_badge = tk.Label(hdr, text='  ⏹ Dừng  ', bg='#45475a',
                                      fg=FG2, font=('Segoe UI', 9), padx=6, pady=2)
        self._status_badge.pack(side='right', padx=10)

        # ── Ngrok token ───────────────────────────────────────────────────────
        tok_fr = tk.Frame(self, bg=DARK, pady=6, padx=12)
        tok_fr.pack(fill='x')
        tk.Label(tok_fr, text='Ngrok Token:', bg=DARK, fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left')
        self._token_var = tk.StringVar(value=self.cfg['ngrok_token'])
        tok_entry = tk.Entry(tok_fr, textvariable=self._token_var, width=52,
                             bg='#313244', fg=FG, insertbackground=FG,
                             relief='flat', font=('Consolas', 9), show='*')
        tok_entry.pack(side='left', padx=6)
        tk.Button(tok_fr, text='👁 Hiện', bg='#45475a', fg=FG, relief='flat',
                  cursor='hand2', font=('Segoe UI', 8),
                  command=lambda: tok_entry.config(
                      show='' if tok_entry.cget('show') else '*')
                  ).pack(side='left', padx=2)
        tk.Button(tok_fr, text='💾 Lưu', bg=ACCENT, fg='#1e1e2e', relief='flat',
                  cursor='hand2', font=('Segoe UI', 8, 'bold'),
                  command=self._save_token).pack(side='left', padx=4)
        self._tok_lbl = tk.Label(tok_fr, text='', bg=DARK, fg=GREEN,
                                 font=('Segoe UI', 8))
        self._tok_lbl.pack(side='left', padx=4)

        # ── Account table ─────────────────────────────────────────────────────
        tbl_fr = tk.Frame(self, bg=DARK, padx=12, pady=4)
        tbl_fr.pack(fill='both', expand=False)
        tk.Label(tbl_fr, text='📋 Tài khoản Kaggle', bg=DARK, fg=ACCENT,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        cols = ('#', 'Gmail', 'Kaggle user', 'Giờ đã dùng', 'Còn lại', 'Trạng thái')
        self._tree = ttk.Treeview(tbl_fr, columns=cols, show='headings', height=6)
        for c, w in zip(cols, [28, 200, 140, 110, 90, 110]):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor='center')
        self._tree.pack(fill='x')
        btn_row = tk.Frame(tbl_fr, bg=DARK, pady=4)
        btn_row.pack(anchor='w')
        for text, cmd in [('➕ Thêm tài khoản', self._add_account),
                          ('✏️ Sửa', self._edit_account),
                          ('🗑 Xóa', self._del_account),
                          ('🔄 Reset giờ tuần này', self._reset_hours)]:
            tk.Button(btn_row, text=text, bg='#313244', fg=FG, relief='flat',
                      cursor='hand2', font=('Segoe UI', 8), padx=8, pady=3,
                      command=cmd).pack(side='left', padx=2)

        # ── Status panel ──────────────────────────────────────────────────────
        st_fr = tk.Frame(self, bg='#181825', padx=12, pady=8)
        st_fr.pack(fill='x')

        row1 = tk.Frame(st_fr, bg='#181825')
        row1.pack(fill='x')
        tk.Label(row1, text='🌐 URL hiện tại:', bg='#181825', fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left')
        self._url_var = tk.StringVar(value='—')
        self._url_entry = tk.Entry(row1, textvariable=self._url_var, width=46,
                                   bg='#313244', fg=GREEN, relief='flat',
                                   font=('Consolas', 9), state='readonly',
                                   readonlybackground='#313244')
        self._url_entry.pack(side='left', padx=6)
        tk.Button(row1, text='📋 Copy', bg='#45475a', fg=FG, relief='flat',
                  cursor='hand2', font=('Segoe UI', 8),
                  command=self._copy_url).pack(side='left', padx=2)

        row2 = tk.Frame(st_fr, bg='#181825', pady=4)
        row2.pack(fill='x')
        tk.Label(row2, text='👤 Tài khoản:', bg='#181825', fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left')
        self._acc_lbl = tk.Label(row2, text='—', bg='#181825', fg=FG,
                                 font=('Segoe UI', 9, 'bold'))
        self._acc_lbl.pack(side='left', padx=8)
        tk.Label(row2, text='⏱ Còn lại hôm nay:', bg='#181825', fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left', padx=(20, 0))
        self._timer_lbl = tk.Label(row2, text='—', bg='#181825', fg=YELLOW,
                                   font=('Consolas', 10, 'bold'))
        self._timer_lbl.pack(side='left', padx=6)
        tk.Label(row2, text='Tuần này còn:', bg='#181825', fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left', padx=(20, 0))
        self._week_lbl = tk.Label(row2, text='—', bg='#181825', fg=FG,
                                  font=('Consolas', 9))
        self._week_lbl.pack(side='left', padx=6)

        row3 = tk.Frame(st_fr, bg='#181825', pady=2)
        row3.pack(fill='x')
        tk.Label(row3, text='Tiến độ 28h:', bg='#181825', fg=FG2,
                 font=('Segoe UI', 9)).pack(side='left')
        self._prog = ttk.Progressbar(row3, length=300, maximum=MAX_HOURS * 3600,
                                     style='TProgressbar')
        self._prog.pack(side='left', padx=8)
        self._prog_lbl = tk.Label(row3, text='0 / 28h', bg='#181825', fg=FG2,
                                  font=('Segoe UI', 8))
        self._prog_lbl.pack(side='left')

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_fr = tk.Frame(self, bg=DARK, pady=8)
        btn_fr.pack()
        self._start_btn = tk.Button(btn_fr, text='▶  Start Server', width=16,
                                    bg=GREEN, fg='#1e1e2e', relief='flat',
                                    cursor='hand2', font=('Segoe UI', 10, 'bold'),
                                    command=self._start)
        self._start_btn.pack(side='left', padx=6)
        self._stop_btn = tk.Button(btn_fr, text='⏹  Stop', width=10,
                                   bg='#45475a', fg=FG, relief='flat',
                                   cursor='hand2', font=('Segoe UI', 10),
                                   command=self._stop, state='disabled')
        self._stop_btn.pack(side='left', padx=6)
        tk.Button(btn_fr, text='🔄 Đổi tài khoản ngay', bg='#45475a',
                  fg=YELLOW, relief='flat', cursor='hand2',
                  font=('Segoe UI', 9), command=self._force_switch).pack(side='left', padx=6)

        # ── Log ───────────────────────────────────────────────────────────────
        log_fr = tk.Frame(self, bg=DARK, padx=12, pady=4)
        log_fr.pack(fill='both', expand=True)
        tk.Label(log_fr, text='📄 Log', bg=DARK, fg=FG2,
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self._log = scrolledtext.ScrolledText(
            log_fr, height=10, bg='#11111b', fg=FG2,
            font=('Consolas', 8), relief='flat', state='disabled',
            insertbackground=FG)
        self._log.pack(fill='both', expand=True)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log_msg(self, msg: str, tag: str = ''):
        ts = datetime.now().strftime('%H:%M:%S')
        self._log.config(state='normal')
        self._log.insert('end', f'[{ts}] {msg}\n')
        self._log.see('end')
        self._log.config(state='disabled')

    def _set_badge(self, text: str, color: str):
        self._status_badge.config(text=f'  {text}  ', bg=color)

    def _copy_url(self):
        url = self._url_var.get()
        if url and url != '—':
            self.clipboard_clear()
            self.clipboard_append(url)
            self._log_msg(f'Đã copy URL: {url}')

    def _save_token(self):
        tok = self._token_var.get().strip()
        self.cfg['ngrok_token'] = tok
        self.cfg.save()
        self._tok_lbl.config(text='✅ Đã lưu')
        self.after(2000, lambda: self._tok_lbl.config(text=''))

    def _restore_url(self):
        if URL_FILE.exists():
            try:
                u = URL_FILE.read_text().strip()
                if u:
                    self._url_var.set(u)
                    self._current_url = u
            except Exception:
                pass

    # ── Account CRUD ──────────────────────────────────────────────────────────
    def _refresh_table(self):
        self.tracker.reset_week_if_needed()
        for row in self._tree.get_children():
            self._tree.delete(row)
        for i, acc in enumerate(self.tracker.accounts(), 1):
            used = acc.get('hours_used', 0)
            remaining = max(0.0, MAX_HOURS - used)
            active = acc.get('active', False)
            status = '● Đang dùng' if active else ('✅ Sẵn sàng' if remaining > 0 else '⛔ Đã hết')
            tag = 'active' if active else ('ok' if remaining > 0 else 'done')
            self._tree.insert('', 'end', iid=str(i-1), values=(
                i, acc.get('gmail', acc['name']), acc['username'],
                f'{used:.1f}h / 28h', f'{remaining:.1f}h', status
            ), tags=(tag,))
        self._tree.tag_configure('active', foreground='#a6e3a1')
        self._tree.tag_configure('done', foreground='#f38ba8')

    def _add_account(self):
        dlg = AccountDialog(self, title='Thêm tài khoản Kaggle')
        self.wait_window(dlg)
        if dlg.result:
            self.cfg['accounts'].append({
                'name': dlg.result['gmail'],
                'gmail': dlg.result['gmail'],
                'username': dlg.result['username'],
                'api_key': dlg.result['api_key'],
                'hours_used': 0.0,
                'week_start': datetime.now().isoformat(),
                'active': False,
            })
            self.cfg.save()
            self._refresh_table()
            self._log_msg(f'Đã thêm: {dlg.result["gmail"]}')

    def _edit_account(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning('', 'Chọn tài khoản cần sửa', parent=self)
            return
        idx = int(sel[0])
        acc = self.tracker.accounts()[idx]
        dlg = AccountDialog(self, title='Sửa tài khoản', existing=acc)
        self.wait_window(dlg)
        if dlg.result:
            acc.update({'gmail': dlg.result['gmail'],
                        'name': dlg.result['gmail'],
                        'username': dlg.result['username'],
                        'api_key': dlg.result['api_key']})
            self.cfg.save()
            self._refresh_table()

    def _del_account(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        acc = self.tracker.accounts()[idx]
        if messagebox.askyesno('Xác nhận', f'Xóa {acc["gmail"]}?', parent=self):
            self.cfg['accounts'].pop(idx)
            self.cfg.save()
            self._refresh_table()

    def _reset_hours(self):
        sel = self._tree.selection()
        if not sel:
            if not messagebox.askyesno('', 'Reset giờ TẤT CẢ tài khoản tuần này?', parent=self):
                return
            for acc in self.tracker.accounts():
                acc['hours_used'] = 0.0
        else:
            idx = int(sel[0])
            acc = self.tracker.accounts()[idx]
            acc['hours_used'] = 0.0
        self.cfg.save()
        self._refresh_table()
        self._log_msg('Đã reset giờ')

    # ── Start / Stop / Switch ─────────────────────────────────────────────────
    def _start(self):
        if not self.cfg['ngrok_token']:
            messagebox.showerror('Lỗi', 'Chưa nhập Ngrok Token!', parent=self)
            return
        if not self.tracker.accounts():
            messagebox.showerror('Lỗi', 'Chưa có tài khoản Kaggle nào!', parent=self)
            return
        acc = self.tracker.best_account()
        if not acc:
            messagebox.showerror('Hết giờ', 'Tất cả tài khoản đã đạt 28h/tuần!\nChờ reset vào thứ Hai.', parent=self)
            return

        if not self._ensure_npoint():
            return

        self._running = True
        self._stop_event.clear()
        self._start_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self._set_badge('⏳ Đang khởi động...', '#f9e2af')

        threading.Thread(target=self._run_session, args=(acc,), daemon=True).start()
        threading.Thread(target=self._timer_loop, daemon=True).start()

    def _stop(self):
        self._running = False
        self._stop_event.set()
        self._set_badge('⏹ Dừng', '#45475a')
        self._start_btn.config(state='normal')
        self._stop_btn.config(state='disabled')
        self._log_msg('Đã dừng.')
        if self._current_account and self._session_start:
            elapsed = (datetime.now() - self._session_start).total_seconds() / 3600
            self.tracker.add_hours(self._current_account['name'], elapsed)
        self._current_account = None
        self._session_start = None
        self._refresh_table()

    def _force_switch(self):
        if not self._running:
            self._start()
            return
        self._log_msg('Đang chuyển tài khoản...')
        if self._current_account and self._session_start:
            elapsed = (datetime.now() - self._session_start).total_seconds() / 3600
            self.tracker.add_hours(self._current_account['name'], elapsed)
        next_acc = self.tracker.best_account()
        if not next_acc:
            messagebox.showerror('', 'Không còn tài khoản nào khả dụng!', parent=self)
            self._stop()
            return
        threading.Thread(target=self._run_session, args=(next_acc,), daemon=True).start()

    # ── Session runner ────────────────────────────────────────────────────────
    def _run_session(self, acc: dict):
        self._current_account = acc
        self._session_start = datetime.now()
        self.tracker.set_active(acc['name'])
        self.after(0, self._refresh_table)
        self._log_msg(f'Bắt đầu session: {acc["gmail"]}')
        self._set_badge('⏳ Đang push kernel...', '#f9e2af')

        script = KERNEL_SCRIPT.replace('{NGROK_TOKEN}', self.cfg['ngrok_token']) \
                               .replace('{NPOINT_ID}', self._npoint_id)

        client = KaggleClient(acc['username'], acc['api_key'])
        if not client.test_auth():
            self.after(0, lambda: messagebox.showerror('Auth lỗi',
                f'Không xác thực được tài khoản {acc["gmail"]}!\nKiểm tra username/API key.', parent=self))
            self._log_msg(f'❌ Auth thất bại: {acc["gmail"]}')
            self._stop()
            return

        result = client.push_kernel(script)
        if 'error' in result:
            self._log_msg(f'❌ Push kernel lỗi: {result["error"]}')
            self._set_badge('❌ Lỗi', '#f38ba8')
            return

        self._log_msg(f'✅ Kernel đã push — đang chờ khởi động (~10 phút)...')
        self._set_badge('⏳ Kaggle đang cài đặt...', '#f9e2af')

        # Chờ URL xuất hiện trên relay
        deadline = datetime.now() + timedelta(minutes=20)
        while datetime.now() < deadline and not self._stop_event.is_set():
            time.sleep(POLL_INTERVAL)
            data = self._poll_relay()
            if data and data.get('status') in ('tunnel_up', 'ready'):
                url = data.get('url', '')
                if url:
                    self._current_url = url
                    self.after(0, lambda u=url: self._update_url(u))
                    self._log_msg(f'✅ URL: {url}')
                    self._set_badge('🟢 Server đang chạy', '#a6e3a1')
                    break
        else:
            if not self._stop_event.is_set():
                self._log_msg('⚠️ Timeout chờ URL — thử lại sau 5 phút')

        # Vòng lặp monitor
        while self._running and not self._stop_event.is_set():
            for _ in range(10):
                if self._stop_event.is_set(): break
                time.sleep(POLL_INTERVAL)

            if self._stop_event.is_set(): break

            elapsed_h = (datetime.now() - self._session_start).total_seconds() / 3600
            remaining_h = self.tracker.hours_remaining(acc['name'])

            # Check relay
            data = self._poll_relay()
            if data:
                url = data.get('url', '')
                status = data.get('status', '')
                if url and url != self._current_url:
                    self._current_url = url
                    self.after(0, lambda u=url: self._update_url(u))
                    self._log_msg(f'URL cập nhật: {url}')
                if status == 'error':
                    self._log_msg('⚠️ Server báo lỗi — có thể đang restart...')
                    self._set_badge('⚠️ Đang restart...', '#f9e2af')
                elif status == 'ready':
                    self._set_badge('🟢 Server đang chạy', '#a6e3a1')

            # Cập nhật giờ
            self.tracker.add_hours(acc['name'], POLL_INTERVAL * 10 / 3600)
            self._session_start = datetime.now()  # reset để tránh double-add

            # Kiểm tra giới hạn 28h
            if remaining_h <= 0.5:
                self._log_msg(f'⚠️ Gần đạt 28h — chuẩn bị chuyển tài khoản!')
                self._set_badge('⚠️ Sắp đổi tài khoản', '#f9e2af')
                time.sleep(60)
                if self._running:
                    self._log_msg('🔄 Chuyển sang tài khoản tiếp theo...')
                    next_acc = self.tracker.best_account()
                    if next_acc and next_acc['name'] != acc['name']:
                        threading.Thread(target=self._run_session,
                                         args=(next_acc,), daemon=True).start()
                    else:
                        self._log_msg('❌ Không còn tài khoản khả dụng!')
                        self._stop()
                break

    def _poll_relay(self) -> dict | None:
        if not self._npoint_id:
            return None
        try:
            r = requests.get(f'https://api.npoint.io/{self._npoint_id}', timeout=8)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    def _update_url(self, url: str):
        self._url_var.set(url)
        URL_FILE.write_text(url)

    # ── Timer loop ────────────────────────────────────────────────────────────
    def _timer_loop(self):
        while self._running and not self._stop_event.is_set():
            acc = self._current_account
            if acc:
                rem = self.tracker.hours_remaining(acc['name'])
                h, m = int(rem), int((rem % 1) * 60)
                total_used = sum(a.get('hours_used', 0) for a in self.tracker.accounts())
                total_rem = max(0, MAX_HOURS * len(self.tracker.accounts()) - total_used)
                used_sec = (MAX_HOURS - rem) * 3600
                self.after(0, lambda r=rem, h_=h, m_=m, u=used_sec, tr=total_rem: (
                    self._timer_lbl.config(text=f'{h_}h {m_:02d}m'),
                    self._acc_lbl.config(text=acc['gmail'] if acc else '—'),
                    self._prog.config(value=min(u, MAX_HOURS * 3600)),
                    self._prog_lbl.config(text=f'{MAX_HOURS - r:.1f} / {MAX_HOURS}h'),
                    self._week_lbl.config(text=f'{tr:.1f}h'),
                    self._refresh_table(),
                ))
            time.sleep(10)

    # ── npoint.io relay ───────────────────────────────────────────────────────
    def _ensure_npoint(self) -> bool:
        if self._npoint_id:
            return True
        self._log_msg('Tạo relay endpoint (npoint.io)...')
        try:
            r = requests.post('https://api.npoint.io/',
                              json={'url': '', 'status': 'idle', 'ts': 0},
                              timeout=10)
            if r.ok:
                nid = r.json().get('id', '')
                if nid:
                    self._npoint_id = nid
                    self.cfg['npoint_id'] = nid
                    self.cfg.save()
                    self._log_msg(f'✅ Relay ID: {nid}')
                    return True
        except Exception as e:
            pass
        messagebox.showerror('Lỗi mạng',
            'Không tạo được relay endpoint.\nKiểm tra kết nối internet.', parent=self)
        return False

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno('Thoát?',
                    'Server đang chạy. Thoát sẽ dừng theo dõi.\nVẫn thoát?', parent=self):
                return
            self._stop()
        self.destroy()


# ─── Account dialog ───────────────────────────────────────────────────────────
class AccountDialog(tk.Toplevel):
    def __init__(self, parent, title='', existing: dict = None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.configure(bg='#1e1e2e')
        self.grab_set()
        self.result = None

        FG, DARK, DARKER = '#cdd6f4', '#1e1e2e', '#313244'

        fields = [
            ('Gmail / Tên hiển thị:', 'gmail', existing.get('gmail', '') if existing else ''),
            ('Kaggle Username:', 'username', existing.get('username', '') if existing else ''),
            ('Kaggle API Key:', 'api_key', existing.get('api_key', '') if existing else ''),
        ]

        tk.Label(self, text='🔑 Thêm tài khoản Kaggle', bg=DARK, fg='#89b4fa',
                 font=('Segoe UI', 11, 'bold'), pady=8).grid(
            row=0, column=0, columnspan=2, sticky='ew', padx=16)

        self._vars = {}
        for i, (label, key, val) in enumerate(fields, 1):
            tk.Label(self, text=label, bg=DARK, fg=FG,
                     font=('Segoe UI', 9), anchor='w').grid(
                row=i, column=0, sticky='w', padx=16, pady=4)
            v = tk.StringVar(value=val)
            show = '*' if key == 'api_key' else ''
            e = tk.Entry(self, textvariable=v, width=36, bg=DARKER, fg=FG,
                         insertbackground=FG, relief='flat',
                         font=('Consolas', 9), show=show)
            e.grid(row=i, column=1, padx=8, pady=4)
            self._vars[key] = v

        tk.Label(self, text='API Key lấy tại: kaggle.com → Account → Create New Token',
                 bg=DARK, fg='#6c7086', font=('Segoe UI', 7)).grid(
            row=4, column=0, columnspan=2, padx=16, pady=2)

        btn_fr = tk.Frame(self, bg=DARK, pady=8)
        btn_fr.grid(row=5, column=0, columnspan=2)
        tk.Button(btn_fr, text='✅ Lưu', bg='#a6e3a1', fg='#1e1e2e',
                  relief='flat', cursor='hand2', font=('Segoe UI', 9, 'bold'),
                  padx=12, command=self._save).pack(side='left', padx=6)
        tk.Button(btn_fr, text='Hủy', bg='#45475a', fg=FG,
                  relief='flat', cursor='hand2', font=('Segoe UI', 9),
                  padx=12, command=self.destroy).pack(side='left', padx=6)

    def _save(self):
        vals = {k: v.get().strip() for k, v in self._vars.items()}
        if not all(vals.values()):
            messagebox.showwarning('', 'Điền đầy đủ thông tin!', parent=self)
            return
        self.result = vals
        self.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.mainloop()
