# CLINC150 intent demo (web app)

A small web app for the CS 72-272 project demo. It runs on any computer (Windows, macOS or Linux) with Python 3.10 or newer, and a CPU is enough.

The page lists the first 10 CLINC150 test sentences, with the outputs of all four methods computed in advance (they are stored in `index.html`). There is also a box where you can type your own request, and typed requests are classified live by every method that is available:

| Card | Where it runs |
|---|---|
| TF-IDF + logistic regression (150 intents, cutoff 0.705) | this computer, if `tfidf_lr_150_msp.joblib` is in this folder |
| RoBERTa-base (150 intents, cutoff 0.79) | this computer (`model/`, `labels.json`) |
| Zero-shot 3B (150 intents, cutoff 0.50) | Chiyu's GPU server (`--zs-url`), needs the API key |
| LoRA 3B (150 intents, cutoff 0.92) | Chiyu's GPU server (`--lora-url`), needs the API key |

A card whose method is not available shows "not connected".

## Model files (not in the repository, too large)

- `model/` (RoBERTa, about 480 MB, required): train the seed-0 150-class model with the code in `../roberta` and copy its `model/` folder here:

      cd ../roberta
      python roberta_clinc.py --mode 150 --out runs/m150_s0 --save_model
      cp -r runs/m150_s0/model ../demo_app/model

- `tfidf_lr_150_msp.joblib` (TF-IDF, about 50 MB, optional): written by Ziyang's TF-IDF script (`tfidf_baseline.py --save-models`, in its `--out` folder). Without it the TF-IDF card shows "not connected".

## Run it

1. Install the packages once:

       python -m venv venv
       venv\Scripts\activate             (Windows)
       source venv/bin/activate          (macOS / Linux)
       pip install torch --index-url https://download.pytorch.org/whl/cpu     (Windows / Linux; on macOS: pip install torch)
       pip install -r requirements.txt

2. Start it and open http://localhost:7862 :

       python app.py

To let other devices on the same network open it (for example a phone, or the classroom computer), start it with `python app.py --host 0.0.0.0` and open `http://<this computer's IP>:7862` on the other device. The system firewall may ask for permission.

## The 3B cards

- Chiyu's two 3B servers run behind temporary tunnels, so their addresses are not stored here. Ask Chiyu for the current addresses and start the app with `python app.py --lora-url https://<address> --zs-url https://<address>`.
- The servers need an API key. Put it in `api_key.txt` in this folder or in the `CLINC_API_KEY` environment variable. Never commit the key (`.gitignore` already leaves `api_key.txt` out).
- A 3B server handles one request at a time, so the app sends it one request at a time.

## Use of generative AI

`app.py`, `index.html` and this README were written by Claude Code (an AI assistant) at Junwei's request. Claude Code also wrote the script that put the precomputed outputs into `index.html` (not included here), and made the TF-IDF model file used in our demo by running Ziyang's `tfidf_baseline.py` unchanged. The models and their outputs come from the team's own code and runs.
