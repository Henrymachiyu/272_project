"""CLINC150 intent demo: a small web app that runs on any computer (Windows, macOS, Linux; CPU is enough).

    pip install -r requirements.txt
    python app.py                    # then open http://localhost:7862
    python app.py --host 0.0.0.0     # also reachable from other devices on the same network
    python app.py --lora-url https://<address> --zs-url https://<address>   # also use Chiyu's 3B servers

The page lists the first 10 CLINC150 test sentences with outputs computed in advance, plus a box to type your
own request. Typed requests are classified live by every method that is available:
  RoBERTa-base   150 intents, cutoff 0.79; runs here (./model, ./labels.json)
  LoRA 3B        150 intents, cutoff 0.92; Chiyu's server (--lora-url), needs the API key in api_key.txt
                 or in the CLINC_API_KEY environment variable
  Zero-shot 3B   150 intents, cutoff 0.50; Chiyu's server (--zs-url), same API key
  TF-IDF         150 intents, cutoff from the model file; runs here if tfidf_lr_150_msp.joblib is in this folder
Methods that are not available show "not connected" on the page.
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBERTA_CUT = 0.79
# Chiyu's 3B servers (150 labels) run behind temporary tunnels, so there is no fixed address here:
# ask Chiyu for the current ones and pass them with --lora-url / --zs-url (empty = card not connected)
LORA_URL = ''
ZS_URL = ''
KEYS = ['tfidf150', 'rB', 'zs150', 'lora150']  # card order on the page
MAX_CHARS = 300


def load_roberta(device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE / 'model')
    model = AutoModelForSequenceClassification.from_pretrained(HERE / 'model').to(device).eval()
    labels = json.loads((HERE / 'labels.json').read_text(encoding='utf-8'))
    lock = threading.Lock()

    def run(text):
        enc = tok([text], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        with lock, torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1)[0].cpu()
        top = torch.argsort(p, descending=True)[:3].tolist()
        conf, intent = float(p[top[0]]), labels[top[0]]
        return dict(final=intent if conf >= ROBERTA_CUT else 'oos', raw=intent, conf=conf, poos=None,
                    oos_wins=False, top3=[[labels[i], float(p[i])] for i in top])
    return run


def load_tfidf(path):
    import joblib
    b = joblib.load(path)
    model, cut = b['model'], float(b['threshold'])
    classes = [str(c) for c in model.classes_]

    def run(text):
        p = model.predict_proba([' '.join(text.lower().split())])[0]
        top = sorted(range(len(p)), key=lambda i: -p[i])[:3]
        conf, intent = float(p[top[0]]), classes[top[0]]
        return dict(final=intent if conf >= cut else 'oos', raw=intent, conf=conf, poos=None,
                    oos_wins=False, top3=[[classes[i], float(p[i])] for i in top])
    return run, cut


def call(base, path, key, body=None, timeout=30):
    req = urllib.request.Request(base.rstrip('/') + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'X-API-Key': key,
                                          'User-Agent': 'clinc-demo'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def connect_3b(base, key, timeout):
    lock = threading.Lock()  # the 3B server handles one request at a time

    def run(text):
        with lock:
            res = call(base, '/api/classify', key, {'text': text, 'generate': False}, timeout)
        top = [[x['label'], float(x['p'])] for x in res['top5'][:3]]
        return dict(final='oos' if res['refused'] else res['prediction'], raw=top[0][0] if top else res['prediction'],
                    conf=float(res['confidence']), poos=res.get('p_oos'), oos_wins=False, top3=top, nogen=True)

    try:  # check the server once at start; the app still starts if it is down
        info = call(base, '/api/info', key, timeout=timeout)
        print(f'  {base}: {info.get("model")}, {info.get("n_labels")} labels, tau {info.get("tau")}')
        if info.get('n_labels') != 150:
            print('  WARNING: this server does not use 150 labels; the page expects the 150-label version')
    except urllib.error.HTTPError as e:
        print(f'  WARNING: {base} answered HTTP {e.code}' + (' (bad API key)' if e.code == 401 else ''))
    except Exception as e:
        print(f'  WARNING: cannot reach {base} ({type(e).__name__}); its card will show "no answer"')
    return run


def explain(e):
    if isinstance(e, urllib.error.HTTPError):
        return 'bad API key' if e.code == 401 else f'HTTP {e.code}'
    if isinstance(e, (urllib.error.URLError, TimeoutError, OSError)):
        return 'server unreachable'
    return type(e).__name__


def make_handler(page, methods):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split('?')[0] in ('/', '/index.html'):
                self.send(200, page, 'text/html; charset=utf-8')
            else:
                self.send(404, b'not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/predict':
                return self.send(404, b'not found', 'text/plain')
            try:
                n = int(self.headers.get('Content-Length', 0))
                text = json.loads(self.rfile.read(min(n, 10000)))['text']
            except Exception:
                return self.send(400, b'bad request', 'text/plain')
            text = ' '.join(str(text).split())[:MAX_CHARS]
            if not text:
                return self.send(400, b'empty request', 'text/plain')
            out, err = {k: None for k in KEYS}, {}
            for k, run in methods.items():
                try:
                    out[k] = run(text)
                except Exception as e:
                    err[k] = explain(e)
                    print(f'{k}: {e!r}')
            self.send(200, json.dumps({'out': out, 'error': err}).encode(), 'application/json')

        def log_message(self, fmt, *args):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description='CLINC150 intent demo web app')
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to let other devices on the network open it')
    ap.add_argument('--port', type=int, default=7862)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--lora-url', default=LORA_URL, help='LoRA 3B server (150 labels); "" to leave it out')
    ap.add_argument('--zs-url', default=ZS_URL, help='zero-shot 3B server (150 labels); "" to leave it out')
    ap.add_argument('--timeout', type=float, default=30, help='seconds to wait for a 3B server')
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress right away, also when output goes to a file

    page = (HERE / 'index.html').read_bytes()
    methods = {}
    print('loading RoBERTa ...')
    methods['rB'] = load_roberta(args.device)
    tfidf_file = HERE / 'tfidf_lr_150_msp.joblib'
    if tfidf_file.exists():
        run, cut = load_tfidf(tfidf_file)
        methods['tfidf150'] = run
        print(f'TF-IDF loaded, cutoff {cut}')
    key_file = HERE / 'api_key.txt'
    key = os.environ.get('CLINC_API_KEY') or (key_file.read_text(encoding='utf-8').strip() if key_file.exists() else '')
    for k, url in (('lora150', args.lora_url), ('zs150', args.zs_url)):
        if url:
            if not key:
                print(f'WARNING: no API key (api_key.txt or CLINC_API_KEY), {k} left out')
                continue
            print(f'connecting {k}:')
            methods[k] = connect_3b(url, key, args.timeout)
    print('connected:', ', '.join(k for k in KEYS if k in methods),
          '| not connected:', ', '.join(k for k in KEYS if k not in methods) or '-')

    server = ThreadingHTTPServer((args.host, args.port), make_handler(page, methods))
    shown = 'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
    print(f'open http://{shown}:{args.port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
