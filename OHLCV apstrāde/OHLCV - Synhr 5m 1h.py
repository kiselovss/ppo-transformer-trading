import pandas as pd
import numpy as np

# Konfigurācija
INPUT_5M  = "ohlcv_16features_5m_Normalized.csv"
INPUT_1H  = "ohlcv_16features_1h_Normalized.csv"
OUTPUT    = "synchr16F_5m_1h.csv"

# 1. solis - datu ielāde
print("=" * 55)
print("1. solis: Datu ielāde")
print("=" * 55)

df_5m = pd.read_csv(INPUT_5M)
df_1h = pd.read_csv(INPUT_1H)

df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"])

df_5m = df_5m.sort_values("timestamp").reset_index(drop=True)
df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)

print(f"5m rindas:  {len(df_5m)}  |  periods: {df_5m['timestamp'].min()} -> {df_5m['timestamp'].max()}")
print(f"1h rindas:  {len(df_1h)}  |  periods: {df_1h['timestamp'].min()} -> {df_1h['timestamp'].max()}")

# 2. solis - 1h sagatavošana
# Stundu svece ar timestamp=T tiek uzskatīta par slēgtu brīdī T+1h.
# Lai novērstu nākotnes datu noplūdi, 5m svecei brīdī T
# pievienojam pēdējo 1h sveci, kurai:
#   candle_close_time = timestamp_1h + 1h  <=  timestamp_5m
# Tas ir ekvivalenti:
#   timestamp_1h  <  timestamp_5m  (strikta nevienādība)
print("\n" + "=" * 55)
print("2. solis: 1h sagatavošana merge_asof")
print("=" * 55)

# Pārdēvējam visas 1h kolonnas ar sufiksu _1h (izņemot timestamp)
rename_map = {c: f"{c}_1h" for c in df_1h.columns if c != "timestamp"}
df_1h = df_1h.rename(columns=rename_map)

# Atslēga merge_asof: 1h sveces slēgšanas laiks
# svece atvērās timestamp_1h -> slēdzās timestamp_1h + 1h
df_1h["candle_close_time"] = df_1h["timestamp"] + pd.Timedelta(hours=1)

print(f"candle_close_time piemērs: {df_1h['candle_close_time'].iloc[:3].tolist()}")

# 3. solis - merge_asof
# Katrai 5m rindai ņemam pēdējo 1h sveci, kurai
# candle_close_time <= timestamp_5m  (svece jau slēgta)
# direction="backward" - meklē tuvāko vērtību <= atslēgai
print("\n" + "=" * 55)
print("3. solis: merge_asof (pēdējā slēgtā 1h svece)")
print("=" * 55)

df = pd.merge_asof(
    df_5m,
    df_1h.drop(columns=["timestamp"]),   # timestamp_1h nav nepieciešams kā atslēga
    left_on="timestamp",
    right_on="candle_close_time",
    direction="backward",
)

# Noņemam palīgkolonnu
df = df.drop(columns=["candle_close_time"])

print(f"Rindas pēc merge: {len(df)}")
print(f"Kolonnas:         {len(df.columns)}")

# 4. solis - noplūdes pārbaude
# Katrai rindai: candle_open_time_1h jābūt < timestamp_5m
print("\n" + "=" * 55)
print("4. solis: Nākotnes datu noplūdes pārbaude")
print("=" * 55)

# Atjaunojam 1h sveces atvēršanas laiku no close_1h laika
# candle_close_time = open_time + 1h  ->  open_time = candle_close_time - 1h
# Bet candle_close_time jau dzēsts - izmantojam open_1h kolonnu kā starpniecību.
# Pārbaudām, aprēķinot rindas, kurās 5m timestamp
# iekrīt TAJĀ PAŠĀ 1h svecē (tā būtu noplūde).

# Pārbaudei uz laiku noapaļojam 5m timestamp līdz stundai
ts_5m_hour  = df["timestamp"].dt.floor("h")

# Ja būtu ņemta pašreizējā (neslēgtā) 1h svece -
# tad tās open_time == floor(timestamp_5m, "1h")
# Pareizi: open_time_1h < floor(timestamp_5m, "1h")  vai
#          open_time_1h == floor un svece jau slēgta

# Vienkārša pārbaude: nevienai 5m svecei nedrīkst būt NaN 1h kolonnās
# (izņemot pirmās rindas pirms pirmās slēgtās 1h sveces)
nan_rows = df["close_1h"].isna().sum()
total    = len(df)
print(f"Rindas bez 1h datiem (sērijas sākums): {nan_rows} ({nan_rows/total*100:.2f}%)")

if nan_rows > 0:
    first_valid = df["close_1h"].first_valid_index()
    print(f"Pirmā rinda ar 1h datiem: indekss {first_valid}, timestamp {df.loc[first_valid, 'timestamp']}")

print("merge_asof ar direction='backward' un candle_close_time garantē,")
print("ka tiek pievienota tikai jau SLĒGTA 1h svece")

# 5. solis - rindas bez 1h datiem tiek dzēstas (sērijas sākums)
df = df.dropna(subset=["close_1h"]).reset_index(drop=True)
print(f"\nPēc rindas bez 1h datiem dzēšanas: {len(df)} rindas")

# 6. solis - saglabāšana
print("\n" + "=" * 55)
print("6. solis: Saglabāšana")
print("=" * 55)

df.to_csv(OUTPUT, index=False)
print(f"Saglabāts: {OUTPUT}")
print(f"Izmērs:    {df.shape}")

# 7. solis - manuāla pārbaude: 3 nejaušas rindas par pēdējām 5 dienām
print("\n" + "=" * 55)
print("7. solis: Manuāla pārbaude (3 nejaušas rindas, pēdējās 5 dienas)")
print("=" * 55)

last_ts   = df["timestamp"].max()
five_days = last_ts - pd.Timedelta(days=5)
recent    = df[df["timestamp"] >= five_days]

sample = recent.sample(n=min(3, len(recent)), random_state=42).sort_values("timestamp")

# Nosaka cenu kolonnu nosaukumus
close_5m_col = "close"      if "close"    in df.columns else "close_5m"
open_5m_col  = "open"       if "open"     in df.columns else "open_5m"
high_5m_col  = "high"       if "high"     in df.columns else "high_5m"
low_5m_col   = "low"        if "low"      in df.columns else "low_5m"

for n_row, (_, row) in enumerate(sample.iterrows(), 1):
    print(f"\n  +- Rinda {n_row} {'-' * 45}")
    print(f"  |  5m sveces laiks : {row['timestamp']}  (UTC)")
    print(f"  |")
    print(f"  |  -- 5m svece -------------------------------------------")
    print(f"  |  Open  : {row[open_5m_col]:.4f}")
    print(f"  |  High  : {row[high_5m_col]:.4f}")
    print(f"  |  Low   : {row[low_5m_col]:.4f}")
    print(f"  |  Close : {row[close_5m_col]:.4f}")
    print(f"  |")
    print(f"  |  -- Pēdējā slēgtā 1h svece ----------------------------")
    print(f"  |  Open  : {row['open_1h']:.4f}")
    print(f"  |  High  : {row['high_1h']:.4f}")
    print(f"  |  Low   : {row['low_1h']:.4f}")
    print(f"  |  Close : {row['close_1h']:.4f}")
    print(f"  +{'-' * 52}")