import ccxt
import pandas as pd
import numpy as np


# 1. Datu ielāde
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=1000):
    exchange = ccxt.binance()

    all_data = []
    since = exchange.parse8601("2019-01-01T00:00:00Z")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_data += ohlcv
        since = ohlcv[-1][0] + 1

        print(f"Ielādēts: {len(all_data)} rindas")

        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


df = fetch_ohlcv()


# 2. ATR - palīgfunkcija, nepieciešama dist_EMA un VWAP_dev aprēķinam
def ATR(df, n=14):
    high_low   = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close  = np.abs(df["low"]  - df["close"].shift())
    tr = np.maximum.reduce([high_low, high_close, low_close])
    return pd.Series(tr, index=df.index).rolling(n).mean()

atr = ATR(df, 14)


# 3. Galīgās pazīmes (16 gabali)

# - Sveču
df["log_return"]       = np.log(df["close"] / df["close"].shift(1))
df["body_pct"]         = (df["close"] - df["open"]) / df["close"]

# - Apjoms
df["vol_ma_ratio"]     = df["volume"] / df["volume"].ewm(span=20).mean()

# - Svārstīgums
df["ATR_norm"]         = atr / df["close"]
df["vol_regime"]       = (df["log_return"].rolling(20).std()
                          / (df["log_return"].rolling(100).std() + 1e-9))

# - Tendence
ema21                  = df["close"].ewm(span=21).mean()
ema55                  = df["close"].ewm(span=55).mean()
df["dist_EMA21"]       = (df["close"] - ema21) / atr
df["dist_EMA55"]       = (df["close"] - ema55) / atr

# - Impulss
def RSI(series, n=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(n).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

df["RSI14"]            = RSI(df["close"], 14)

# - VWAP novirze
pv                     = df["close"] * df["volume"]
vwap                   = pv.rolling(20).sum() / df["volume"].rolling(20).sum()
df["VWAP_dev"]         = (df["close"] - vwap) / atr

# - Laiks / sesija
hour                   = df["timestamp"].dt.hour
df["hour_sin"]         = np.sin(2 * np.pi * hour / 24)
df["hour_cos"]         = np.cos(2 * np.pi * hour / 24)
df["is_asia"]          = (hour < 8).astype(int)
df["is_europe"]        = ((hour >= 8) & (hour < 16)).astype(int)
df["is_us"]            = (hour >= 16).astype(int)

# - Papildu
df["range_norm"]       = (df["high"] - df["low"]) / atr
df["trend_consistency"]= np.sign(df["log_return"]).rolling(10).mean()


# Tīrīšana
# Atstājam tikai pamata OHLCV + 16 pazīmes
KEEP_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    # sveču
    "log_return", "body_pct",
    # apjoms
    "vol_ma_ratio",
    # svārstīgums
    "ATR_norm", "vol_regime",
    # tendence
    "dist_EMA21", "dist_EMA55",
    # impulss
    "RSI14",
    # VWAP
    "VWAP_dev",
    # laiks / sesija
    "hour_sin", "hour_cos", "is_asia", "is_europe", "is_us",
    # papildu
    "range_norm", "trend_consistency",
]

df = df[KEEP_COLS].dropna().reset_index(drop=True)

print(df.head())
print(f"\nIzmērs: {df.shape}")
print(f"Kolonnas ({len(df.columns)}): {df.columns.tolist()}")

df.to_csv("ohlcv_16features_1h.csv", index=False)