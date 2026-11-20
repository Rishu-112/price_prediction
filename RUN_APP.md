# Running the Stock Price Prediction App

Local setup notes for `app_final.py` (Streamlit app) on Windows.

## Requirements

- Python **3.11** (`tensorflow==2.15.0` does not support 3.12 / 3.13)
- Internet connection (stock data is downloaded from Yahoo Finance on each run)

Install Python 3.11 if needed:

```powershell
winget install -e --id Python.Python.3.11 --scope user
```

## One-time setup

The virtual environment is created at a **short path outside the project**
(`C:\venvs\stock`). Creating it inside the project/OneDrive folder fails on
Windows because TensorFlow's file paths exceed the path-length limit. This also
keeps the venv out of git.
"C:\venvs\stock\Scripts\Activate.ps1"

```powershell
cd "C:\Users\rishu\OneDrive\Documents\Desktop\Stock-Price-Prediction-main\Stock-Price-Prediction-main"

New-Item -ItemType Directory -Force C:\venvs
py -3.11 -m venv C:\venvs\stock
C:\venvs\stock\Scripts\python.exe -m pip install -r requirements.txt
```

The install is large (~1.5-2 GB) and takes several minutes.

## Run the app

```powershell
cd "C:\Users\rishu\OneDrive\Documents\Desktop\Stock-Price-Prediction-main\Stock-Price-Prediction-main"
C:\venvs\stock\Scripts\streamlit.exe run app_final.py
```

Then open <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop it.

Optional: skip the first-run email prompt / open no browser:

```powershell
C:\venvs\stock\Scripts\streamlit.exe run app_final.py --server.headless true
```

Alternative (activate the venv first):

```powershell
C:\venvs\stock\Scripts\Activate.ps1
streamlit run app_final.py
```

## Run the notebook

`Stock_Market_Prediction_Model_Creation.ipynb` retrains the LSTM model.
Note: it **overwrites** `Stock Prediction.keras` and `predicted_stock_prices.csv`
(the model the app loads), so back them up first if you want to keep the originals.

```powershell
C:\venvs\stock\Scripts\python.exe -m pip install jupyter ipykernel
C:\venvs\stock\Scripts\python.exe -m ipykernel install --user --name stock311
C:\venvs\stock\Scripts\python.exe -m jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
```

Pick the `stock311` kernel in Jupyter.

## Notes

- `app_final.py` loads `Stock Prediction.keras` from the current directory, so
  run the command from the project folder.
- Do not commit the venv or push this setup to git. `.gitignore` already lists `venv/`.
