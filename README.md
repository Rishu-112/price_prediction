# 📈 Stock Price Prediction using LSTM

This project leverages a **Long Short-Term Memory (LSTM)** neural network to predict future stock prices. It provides an interactive web interface built with **Streamlit**, allowing users to visualize historical trends, compare model predictions against actual prices, and forecast future stock movements.

## 🚀 Features

- **Real-Time Data Fetching**: Integrates with `yfinance` to pull the latest stock market data.
- **Interactive Technical Analysis**: 
  - Candlestick charts for historical price action.
  - Volume bars to analyze market activity.
  - Moving Averages (**MA50** and **MA200**) to identify trends.
- **Model Evaluation**: Calculates and displays **Model Accuracy (MAPE)** and **Root Mean Squared Error (RMSE)** to quantify prediction reliability.
- **Future Forecasting**: Predicts the stock price for the next **100 business days** based on learned patterns.
- **Interactive Visuals**: High-quality, zoomable charts powered by **Plotly**.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [Keras / TensorFlow](https://keras.io/) (LSTM Architecture)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Financial Data**: [yfinance](https://pypi.org/project/yfinance/)
- **Visualization**: [Plotly](https://plotly.com/)
- **Preprocessing**: [Scikit-learn](https://scikit-learn.org/) (MinMaxScaler)

## ⚙️ Installation & Setup

> **Python version:** `tensorflow==2.15.0` supports Python **3.9 – 3.11** only (not 3.12 / 3.13). Python 3.11 is recommended.

### Option A: Windows (venv, as used in this setup)

Create the virtual environment at a **short path outside the project**
(e.g. `C:\venvs\stock`). Creating it inside a deep folder such as OneDrive/Desktop
can fail on Windows because TensorFlow's file paths exceed the path-length limit.
It also keeps the venv out of git.

```powershell
# 1. Install Python 3.11 (if needed)
winget install -e --id Python.Python.3.11 --scope user

# 2. Create the environment and install dependencies
cd <path-to-this-project-folder>
New-Item -ItemType Directory -Force C:\venvs
py -3.11 -m venv C:\venvs\stock
C:\venvs\stock\Scripts\python.exe -m pip install -r requirements.txt
```

The install is large (~1.5–2 GB) and can take several minutes.

**Run the app** (from the folder containing `app_final.py`):

```powershell
C:\venvs\stock\Scripts\streamlit.exe run app_final.py
```

Then open <http://localhost:8501>. Or activate the environment first:

```powershell
C:\venvs\stock\Scripts\Activate.ps1
streamlit run app_final.py
```

### Option B: Conda (macOS / Linux / Windows)

```bash
conda create -n venv python=3.10 -y
conda activate venv
pip install -r requirements.txt
streamlit run app_final.py
```

## 📓 Running the Notebook

`Stock_Market_Prediction_Model_Creation.ipynb` trains the LSTM model and saves it as
`Stock Prediction.keras` (plus `predicted_stock_prices.csv`). Running it **overwrites**
those two files, so back them up if you want to keep the originals.

```powershell
C:\venvs\stock\Scripts\python.exe -m pip install jupyter ipykernel
C:\venvs\stock\Scripts\python.exe -m ipykernel install --user --name stock311
C:\venvs\stock\Scripts\python.exe -m jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
```

Select the **`stock311`** kernel (in VS Code: *Select Kernel → Jupyter Kernel → stock311*).

> If the price-chart cell fails with `Invalid property ... titlefont_size`, change
> `titlefont_size=14` to `title_font_size=14` (newer Plotly versions).

## 🧩 Troubleshooting

- **`C:\venvs\stock` "is not recognized"**: it is a folder, not a command. Run
  `C:\venvs\stock\Scripts\streamlit.exe ...` or activate with `Activate.ps1`.
- **`Activate.ps1` blocked**: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Model not found**: run the app from the folder that contains `Stock Prediction.keras`.
- **No data / empty charts**: the app downloads data from Yahoo Finance on every run, so it needs internet access.

## 📊 How it Works
1. **Data Collection**: The app fetches historical data for a chosen stock symbol.
2. **Preprocessing**: Data is normalized using `MinMaxScaler` to help the LSTM model converge faster.
3. **LSTM Prediction**: The model uses a sliding window of the previous 100 days to predict the price for the next day.
4. **Future Loop**: To predict 100 days into the future, the model uses its own previous prediction as input for the next day (recursive forecasting).

