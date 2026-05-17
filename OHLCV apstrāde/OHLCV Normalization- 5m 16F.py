import pandas as pd
import numpy as np

# Konfigurācija
INPUT_FILE  = "ohlcv_16features_5m.csv"
OUTPUT_FILE = "ohlcv_16features_5m_Normalized.csv"

EPS = 1e-9

# Ielāde
df = pd.read_csv(INPUT_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Ielādētas rindas:   {len(df)}")
print(f"Ielādētas kolonnas: {len(df.columns)}")

# Normalizācija
#
# Pilnvērtīga normalizācija (2 pazīmes):
#   vol_ma_ratio  - labās puses izmērs -> clip(0,5) + rolling min-max
#   RSI14         - diapazons [0,100], nesimetrisks -> (RSI-50)/50
#
# Aizsargājošs clip pret astes vērtībām (4 pazīmes):
#   dist_EMA21    - normalizēts pēc ATR, bet stiprās tendencēs aiziet tālu
#   dist_EMA55    - tas pats, pēdējās rindās redzētas vērtības 4+
#   VWAP_dev      - normalizēts pēc ATR, līdzīgs izmēru raksturs
#   range_norm    - vienmēr >=0, bet bez augšējās robežas svārstīgās dienās
#
# Netiekam klāt (jau vajadzīgajā diapazonā):
#   log_return        - bezizmēra un simetriska
#   body_pct          - diapazons aptuveni (-1, 1)
#   ATR_norm          - bezizmēra
#   vol_regime        - attiecība, clip(0.2, 5) pēc izvēles
#   hour_sin          - jau [-1, 1]
#   hour_cos          - jau [-1, 1]
#   is_asia           - binārā {0, 1}
#   is_europe         - binārā {0, 1}
#   is_us             - binārā {0, 1}
#   trend_consistency - jau [-1, 1]
#
# Visas rolling operācijas skatās tikai atpakaļ - nākotnes noplūdes nav.

# 1. vol_ma_ratio - clip(0, 5) + rolling min-max (window=200)
#    Iemesls: labās puses izmērs apjomīgos impulsos
clipped            = df["vol_ma_ratio"].clip(0, 5)
vr_roll_min        = clipped.rolling(200, min_periods=1).min()
vr_roll_max        = clipped.rolling(200, min_periods=1).max()
df["vol_ma_ratio"] = (clipped - vr_roll_min) / (vr_roll_max - vr_roll_min + EPS)

print("✓ vol_ma_ratio - clip(0, 5) + rolling min-max(200)")

# 2. RSI14 - (RSI - 50) / 50 -> diapazons [-1, 1]
#    Iemesls: sākotnējais diapazons [0, 100], nesimetrisks
#    Konstantā transformācija - noplūdes nav pēc definīcijas
df["RSI14"] = (df["RSI14"] - 50) / 50

print("✓ RSI14 - (RSI - 50) / 50")

# 3. dist_EMA21, dist_EMA55, VWAP_dev - clip(-5, 5)
#    Iemesls: normalizēti pēc ATR, bet stiprās tendencēs aiziet aiz +-4
#    Clip nemaina sadalījumu normā, tikai apgriež retās astes
for col in ["dist_EMA21", "dist_EMA55", "VWAP_dev"]:
    df[col] = df[col].clip(-5, 5)

print("✓ dist_EMA21, dist_EMA55, VWAP_dev - clip(-5, 5)")

# 4. range_norm - clip(0, 5)
#    Iemesls: vienmēr >=0 pēc definīcijas, bet ekstrēmi
#    svārstīgās dienās (flash crash, likvidācijas) var dot izmērus
df["range_norm"] = df["range_norm"].clip(0, 5)

print("✓ range_norm - clip(0, 5)")

# Diapazona pārbaude
print("\n--- Diapazona pārbaude pēc normalizācijas ---")

check_cols = {
    "vol_ma_ratio": (0,  1),
    "RSI14":        (-1, 1),
    "dist_EMA21":   (-5, 5),
    "dist_EMA55":   (-5, 5),
    "VWAP_dev":     (-5, 5),
    "range_norm":   (0,  5),
}
all_ok = True
for col, (lo, hi) in check_cols.items():
    actual_min = df[col].min()
    actual_max = df[col].max()
    ok = actual_min >= lo - 0.01 and actual_max <= hi + 0.01
    status = "✓" if ok else "✗"
    print(f"  {status} {col}: [{actual_min:.4f}, {actual_max:.4f}]  (gaidīts [{lo}, {hi}])")
    if not ok:
        all_ok = False

if all_ok:
    print("\nVisas pārbaudes veiksmīgas")
else:
    print("\nIr novirzes - nepieciešams pārbaudīt loģiku augstāk")

# Saglabāšana
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaglabāts: {OUTPUT_FILE}")
print(f"Izmērs:    {df.shape}")
print(f"Kolonnas:  {df.columns.tolist()}")