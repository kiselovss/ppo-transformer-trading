"""
PPO Transformer Tirdzniecības Aģents — Clean Baseline režīms
Pastiprinājuma Mācīšanās, Aktieris-Kritiķis, Almgren-Chriss Slīdēšana


APRAKSTS
────────
Aģents tiek apmācīts ar Proximal Policy Optimization (PPO) metodi uz
kriptovalūtu tirgus datiem (5 minūšu sveces). Kā policy tiek izmantots
Transformer-enkodētājs: modelis redz slīdošu logu no SEQ_LEN pēdējām svecēm
un pieņem vienu no četriem lēmumiem - HOLD, LONG, SHORT, CLOSE.

Slīdēšanas modelis ir veidots uz Almgren-Chriss ar trim tirgus režīmiem
(normal / elevated / stress) un saglabā reālistiskas tirdzniecības izmaksas
pat pie nelieliem pozīcijas apjomiem.

CLEAN BASELINE REŽĪMS
─────────────────────
Šis ir pamata kontroles eksperiments bez reward shaping soļiem.
Visi sodi un stimuli ir atslēgti:
  - Nav atvēršanas soda (OPEN_PENALTY = 0.0)       aģents var brīvi tirgoties
  - Nav HOLD soda (HOLD_PENALTY_NO_POS = 0.0)       bezdarbība netiek sodīta
  - Nav īsā darījuma soda (SHORT_TRADE_PEN = 0.0)     skalping atļauts
  - Nav minimālā darījuma ilguma (MIN_TRADE_BARS = 0)
  - Nav labvēlības perioda (GRACE_PERIOD_STEPS = 0)
  - Nav drawdown soda (DD_COEF = 0.0)

Apmācības laikā tiek izmantots tikai tīrs PnL signāls (log-atgriešanās).
Tas ļauj pārbaudīt, vai datos vispār ir izmantojams signāls pirms
reward shaping pievienošanas.

IZMANTOŠANA
───────────
Salīdzinu bektesta rezultātus ar Realistic Mode:
Ja Clean Baseline ir sliktāks, tad reward shaping palīdz
Ja Clean Baseline ir labāks, tad iespējams, sodi ir pārāk agresīvi
Bektests vienmēr notiek ar pilnām komisijām un slīdēšanu


────────────────────────────────────────────────────────────────────────────────
VIDES PRASĪBAS
────────────────────────────────────────────────────────────────────────────────
  Python       : 3.10 - 3.12  (ieteicams 3.11)
  PyTorch      : >= 2.1  (ar CUDA 12.x atbalstu GPU paātrinājumam)
  Galvenās pakotnes:
    numpy        >= 1.26
    pandas       >= 2.1
    scikit-learn >= 1.4    (StandardScaler)
    torch        >= 2.1

  Atkarību uzstādīšana:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scikit-learn

────────────────────────────────────────────────────────────────────────────────
APARATŪRAS RESURSI
────────────────────────────────────────────────────────────────────────────────
  GPU (ieteicams):
    Testēts uz Google Colab T4 (15 GB VRAM), apmācība ar AMP,
      aptuveni 2-8 min uz 75 000 soļiem pie pašreizējās arhitektūras.
    Ieslēdzas automātiski (torch.cuda.is_available()).

  CPU:
    Testēts uz AMD Ryzen 3 1200 Quad-Core, apmācība bez AMP,
      aptuveni 30-60 min uz 75 000 soļiem.

  RAM: minimums 4 GB; ieteicami 8 GB pie lielām datu kopām.

────────────────────────────────────────────────────────────────────────────────
IEEJAS DATI
────────────────────────────────────────────────────────────────────────────────
  Fails: tiek iestatīts ar mainīgo INPUT_FILE (pēc noklusējuma "synchr16F_5m_1h.csv").
  CSV formāts ar kolonnām:
    timestamp, open, high, low, close, volume      - obligātās
    open_1h, high_1h, low_1h, close_1h, volume_1h - stundas sveces (pēc izvēles)
    ... jebkuri skaitliski pazīmju lauki (tehniskie indikatori, pazīmes)
  Kolonnas ar prefiksu "target_" tiek automātiski izslēgtas no pazīmēm.
  Dati tiek sadalīti: pēdējā DATA_WINDOW_FRACTION daļa no visas datu kopas,
  no tās 2/3 - apmācība, 1/3 - tests.

────────────────────────────────────────────────────────────────────────────────
IZEJAS FAILI
────────────────────────────────────────────────────────────────────────────────
  ppo_trader_{VERSION}_{SYMBOL}.log       - pilns apmācības un bektesta žurnāls
  ppo_trader_{VERSION}_{SYMBOL}.pth       - galīgā modeļa svari
  backtest_{VERSION}_{SYMBOL}_trades.csv  - visu darījumu žurnāls (bektests scen.1)
  experiments.csv                          - uzkrājošais eksperimentu žurnāls
  checkpoints_{VERSION}_{SYMBOL}/         - starpsavienojuma modeļa kontrolpunkti
"""

import os, sys, math, random, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")


# Versija un tirgus simbols:
# Šie divi mainīgie parādās visu izejas failu nosaukumos:
#   žurnālu, saglabātā modeļa, CSV darījumu un kontrolpunktu direktorijas.
# SYMBOL var mainīt, lai nomainītu tirgu (BTC, ETH, SOL, utt.)
# VERSION - pašreizējās implementācijas vai apmācības režīma nosaukums.
VERSION = "Clean Baseline"
SYMBOL  = "BTC"


# Reģistrētājs:
# TeeLogger dublē visu izvadi vienlaicīgi konsolē un žurnāla failā.
# Tas ļauj reproducēt apmācības vēsturi pēc skripta pabeigšanas
# (svarīgi pie ilgas apmācības uz attālā Colab servera).

LOG_FILE = f"ppo_trader_{VERSION}_{SYMBOL}.log"

class TeeLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

import atexit
_logger_instance = TeeLogger(LOG_FILE)
sys.stdout = _logger_instance
atexit.register(_logger_instance.close)
print(f"Žurnāls: {LOG_FILE}\n")


# Konfigurācija:
# Globālie modeļa, apmācības un datu hiperparametri.
# Arhitektūra ir apzināti kompakta (D_MODEL=64, N_LAYERS=1)
# tas dod aptuveni 4 reizes mazāk parametru salīdzinājumā ar agrīnām versijām
# ievērojami paātrina apmācību bez manāma kvalitātes zuduma.
INPUT_FILE = f"{SYMBOL}_synchr16F_5m_1h.csv"

# Izmantojam pēdējo 3/7 datu kopas daļu (aptuveni 3 gadi): 2 gadi apmācībai, 1 gads testam.
# Šī vērtība tika izvēlēta fiksēta sadalījuma vietā
# Ļauj ņemt aktuālus datus un nepārmācīties uz novecojušiem tirgus režīmiem.
DATA_WINDOW_FRACTION = 3 / 7
TRAIN_FRACTION       = 2 / 3
SEED                 = 42

# Transformer enkodētāja parametri
SEQ_LEN   = 32    # novērošanas loga garums (sveces)
D_MODEL   = 64    # iegulšanas dimensionalitāte
N_HEADS   = 2     # pašuzmanības galvu skaits
N_LAYERS  = 1     # Transformer bloku skaits
DROPOUT   = 0.1
N_ACTIONS = 4     # 0=HOLD  1=LONG  2=SHORT  3=CLOSE

# PPO un apmācības hiperparametri.
# TOTAL_STEPS=75_000 ir pietiekami konverģencei pie pašreizējās arhitektūras.
N_ENVS        = 4      # paralēlo vižu skaits (diverse starting points)
ROLLOUT_STEPS = 128    # soļi uz rollout katrā vidē
MINI_BATCH    = 128    # mini-paketes izmērs PPO atjauninājumam
PPO_EPOCHS    = 2      # paketes caurlaižu skaits vienam atjauninājumam
TOTAL_STEPS   = 75_000
CLIP_EPS      = 0.2    # varbūtību attiecības apgriešana (PPO standarts)
LR            = 3e-4
GAMMA         = 0.99   # diskonta faktors
GAE_LAM       = 0.95   # λ Vispārinātajam Priekšrocību Novērtējumam (GAE)
ENT_COEF      = 0.05   # entropijas bonusa koeficients (izpēte)
V_COEF        = 1.0    # vērtības zuduma koeficients
MAX_GRAD_NORM = 0.5    # gradienta apgriešana


# Eksperimenta bloks — Clean Baseline:
# Visi reward shaping parametri ir atslēgti (nulle).
# Aģents māca tikai no tīra PnL signāla — log-atgriešanās no sveces uz sveci.
# Nav nekādu sodu vai stimulu, kas ietekmētu tirdzniecības biežumu vai ilgumu.
#
# Mērķis: pārbaudīt, vai tirgus datos ir izmantojams signāls vispār,
# pirms pievienot reward shaping. Šis rezultāts kalpo kā kontroles grupa
# salīdzinājumā ar Basic Mode un citiem eksperimentālajiem režīmiem.
EXPERIMENT = {
    "name"               : "Clean Baseline",
    "SEED"               : SEED,
    "TOTAL_STEPS"        : TOTAL_STEPS,

    # Visi sodi atslēgti — tīrs signāls bez reward shaping.
    "HOLD_PENALTY_NO_POS": 0.0,    # nav soda par bezdarbību (Basic Mode: -0.003)
    "INVALID_ACTION_PEN" : -0.05,  # nederīga darbība joprojām sodīta (loģikas aizsardzība)
    "REWARD_CLIP"        : 0.3,    # reward ierobežojums saglabāts kā Basic Mode
    "DD_COEF"            : 0.0,    # drawdown sods atslēgts (tāpat kā Basic Mode)

    # Labvēlības periods atslēgts — nav nepieciešams bez HOLD soda.
    "GRACE_PERIOD_STEPS" : 0,

    # Galvenie reward shaping parametri — visi nulle.
    # Salīdzinājumam: Basic Mode izmanto OPEN_PENALTY=0.000898.
    "OPEN_PENALTY"       : 0.0,    # nav soda par darījuma atvēršanu

    # Minimālais darījuma ilgums atslēgts — skalping nav ierobežots.
    # Salīdzinājumam: Basic Mode izmanto MIN_TRADE_BARS=72.
    "MIN_TRADE_BARS"     : 0,

    # Anti-skalping sods atslēgts kopā ar MIN_TRADE_BARS.
    "SHORT_TRADE_PEN"    : 0.0,
}

# Darījumu izmaksu parametri:
# COMMISSION_PCT - simetriska komisija ieejai un izejai.
# SLIPPAGE_PARAMS - Almgren-Chriss modelis: ņem vērā pusi spredas,
# tirgus ietekmi (impact), svārstīgumu un nejaušu troksni.
# Piezīme: izmaksas tiek lietotas TIKAI bektestā, nevis apmācības laikā.
# Apmācībā komisija un slīdēšana ir ieslēgtas (use_commission=True, use_slippage=True),
# bet bez reward shaping sodiem to ietekme uz aģenta uzvedību ir minimāla.
COMMISSION_PCT = 0.0004   # 0.04% no katras puses (kā Binance futures)

SLIPPAGE_PARAMS = {
    "half_spread"   : 0.00010,   # puse no bid-ask spredas
    "impact_alpha"  : 0.10,      # lineārais tirgus ietekmes koeficients
    "impact_beta"   : 0.01,      # kvadrātiskais ietekmes koeficients
    "vol_gamma"     : 0.30,      # svārstīguma ieguldījums slīdēšanā
    "noise_delta"   : 0.10,      # nejaušā komponente (σ * noise_delta)
    "buy_asymmetry" : 1.15,      # pirkumi dārgāki par pārdošanu (asimetriska likviditāte)
    "min_slippage"  : 0.00005,   # minimālā slīdēšanas grīda
    "vol_window"    : 20,        # logs slīdošajam svārstīgumam
}

# Tirgus režīma pārslēgšanas sliekšņi (pēc log-atgriešanās slīdošā svārstīguma).
# Pārsniedzot sliekšņus, slīdēšana tiek mērogota ar režīma reizinātāju.
REGIME_THRESHOLDS  = {"normal": 0.0010, "elevated": 0.0025, "stress": 0.0050}
REGIME_MULTIPLIERS = {"normal": 1.0,    "elevated": 2.0,    "stress": 3.5}

# 5 minūšu sveces skaits gadā: 252 tirdzniecības dienas × 24 h × 12 sveces/h.
# Izmanto gadskārtējā Sharpe aprēķinā caur darījumu biežumu.
BARS_PER_YEAR = 252 * 24 * 12

# Aprēķinu ierīce un Jaukta Precizitāte (AMP).
# AMP ieslēdzas tikai pie CUDA esamības un paātrina GPU apmācību aptuveni 1.5-2 reizes.
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Ierīce: {DEVICE}  |  AMP: {USE_AMP}")
print(f"Eksperiments: {EXPERIMENT['name']}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 1. solis - Datu ielāde un sagatavošana:
print("\n" + "="*60)
print("1. SOLIS: Datu ielāde")
print("="*60)

df_full = pd.read_csv(INPUT_FILE)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_full = df_full.sort_values("timestamp").reset_index(drop=True)

# Ņemu tikai pēdējo vēstures daļu (DATA_WINDOW_FRACTION).
# Lai apmācītos uz aktuāliem datiem, nevis uz tirgus režīmiem
# pirms 5-7 gadiem, kas var neatbilst pašreizējiem apstākļiem.
window_start = int(len(df_full) * (1 - DATA_WINDOW_FRACTION))
df = df_full.iloc[window_start:].reset_index(drop=True)
split_idx = int(len(df) * TRAIN_FRACTION)

print(f"Visa datu kopa : {len(df_full):,} sveces  |  {df_full['timestamp'].min()} → {df_full['timestamp'].max()}")
print(f"Izmantojam     : {len(df):,} sveces  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Apmācība (~2g) : 0..{split_idx:,}  ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[split_idx-1]})")
print(f"Tests  (~1g)   : {split_idx:,}..{len(df):,}  ({df['timestamp'].iloc[split_idx]} → {df['timestamp'].iloc[-1]})")

# Automātiski izvēlamies visas skaitliskās pazīmes, izņemot OHLCV kolonnas un mērķus.
# Šāda pieeja ļauj pievienot jaunus indikatorus CSV bez koda izmaiņām.
NON_FEATURE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume",
    "open_1h", "high_1h", "low_1h", "close_1h", "volume_1h",
}
feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURE_COLS
    and not c.startswith("target_")
    and pd.api.types.is_numeric_dtype(df[c])
]
print(f"Pazīmes (tirgus): {len(feature_cols)}")

close_prices = df["close"].values.astype(np.float32)
open_prices  = df["open"].values.astype(np.float32)
high_prices  = df["high"].values.astype(np.float32)
low_prices   = df["low"].values.astype(np.float32)
volumes      = (df["volume"].values.astype(np.float32)
                if "volume" in df.columns else np.ones(len(df), np.float32))
timestamps   = df["timestamp"].values

X_raw = df[feature_cols].values.astype(np.float32)
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Svarīgi: StandardScaler tiek apmācīts tikai uz apmācības daļas, pēc tam lietots testam.
# Tas novērš datu noplūdi (agrīnās koda versijās scaler tika apmācīts uz visas datu kopas,
# kas deva negodīgu priekšrocību).
scaler = StandardScaler()
X = X_raw.copy()
X[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
X[split_idx:] = scaler.transform(X_raw[split_idx:])

# Log-atgriešanās izmanto compute_slippage slīdošā svārstīguma aprēķinam.
log_returns = np.zeros(len(X), dtype=np.float32)
log_returns[:-1] = np.log(
    (close_prices[1:] + 1e-9) / (close_prices[:-1] + 1e-9)
)

# N_STATE_FEATURES = tirgus pazīmes + position + unrealized_pnl.
# Divas papildu pazīmes dod aģentam tiešas zināšanas par pašreizējo pozīciju.
N_MARKET_FEATURES = X.shape[1]
N_STATE_FEATURES  = N_MARKET_FEATURES + 2
INPUT_DIM         = N_STATE_FEATURES
print(f"N_STATE_FEATURES={N_STATE_FEATURES}")


# 2. solis - Slīdēšanas modelis (Almgren-Chriss)
# Reālistisks darījumu izmaksu novērtējums, ņemot vērā pašreizējo tirgus režīmu.
# Rindu priekšaprēķins (rolling_vol, ATR utt.)
# tiek veikts vienreiz, lai nepalēninātu simulāciju step() iekšienē.

def precompute_market_series(close, high, low, volumes, log_ret,
                             vol_window=20, train_end_idx=None):
    """
    Priekšaprēķina palīglaikrindu sērijas slīdēšanas aprēķinam:
      rolling_vol  - slīdošais svārstīgums (log-atgriešanās std)
      volume_usd   - apjoms USD (izmanto tirgus ietekmes novērtēšanai)
      atr          - Vidējais Patiesais Diapazons (14-periodu)
    """
    n           = len(close)
    rolling_vol = np.full(n, SLIPPAGE_PARAMS["min_slippage"], dtype=np.float32)
    for i in range(vol_window, n):
        rolling_vol[i] = float(np.std(log_ret[i - vol_window:i]))
    rolling_vol = np.maximum(rolling_vol, 1e-6)
    volume_usd  = close * volumes
    atr = np.full(n, 0.001 * close[0], dtype=np.float32)
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i]  - close[i-1]))
        atr[i] = (atr[i-1] * 13 + tr) / 14 if i >= 14 else tr
    return {"rolling_vol": rolling_vol, "volume_usd": volume_usd,
            "atr": atr, "high": high, "low": low}

print("\nSlīdēšanas rindu priekšaprēķins...")
market_series = precompute_market_series(
    close_prices, high_prices, low_prices, volumes, log_returns,
    train_end_idx=split_idx)


def compute_slippage(i, direction, balance, price,
                     rng=None, p=SLIPPAGE_PARAMS, series=market_series):
    """
    Aprēķina galīgo slīdēšanu pēc Almgren-Chriss modeļa.
    Veido: half_spread + ietekme (α·σ·√Q/V + β·σ·Q/V) + vol_term + troksnis.
    Palielinās 2 reizes pie elevated un 3.5 reizes pie stress svārstīguma režīma.
    Pirkumi dārgāki par pārdošanu (buy_asymmetry=1.15, likviditātes asimetrija).
    Ierobežots ar pusi sveces diapazona (reālistisks griests).
    """
    σ = float(series["rolling_vol"][i])
    V = float(series["volume_usd"][i])
    Q = float(balance)
    noise = (rng.normal(0, p["noise_delta"] * σ) if rng is not None
             else np.random.normal(0, p["noise_delta"] * σ))
    slip = max(
        p["half_spread"]
        + p["impact_alpha"] * σ * np.sqrt(Q / (V + 1.0))
        + p["impact_beta"]  * σ * (Q / (V + 1.0))
        + p["vol_gamma"]    * σ
        + noise,
        0.0
    )
    if σ >= REGIME_THRESHOLDS["stress"]:
        slip *= REGIME_MULTIPLIERS["stress"]
    elif σ >= REGIME_THRESHOLDS["elevated"]:
        slip *= REGIME_MULTIPLIERS["elevated"]
    if direction > 0:
        slip *= p["buy_asymmetry"]
    slip = max(slip, p["min_slippage"])
    candle_range = abs(series["high"][i] - series["low"][i]) / (price + 1e-9)
    if candle_range > 0:
        slip = min(slip, candle_range * 0.5)
    return float(slip), float(price * (1.0 + direction * slip))


# 3. solis - Darbības maskēšana (Action Masking)
# Aizliedz nederīgas darbības politikas līmenī, nevis caur sodu.
# Agrīnās versijās nederīgas darbības vienkārši tika ignorētas, kas maldināja
# aģentu apmācības laikā.

def get_valid_mask(position):
    """
    Atgriež pieļaujamo darbību Būla masku pašreizējai pozīcijai.
    0=HOLD  1=LONG  2=SHORT  3=CLOSE
    CLOSE ir aizliegts, ja nav atvērtas pozīcijas (position == 0).
    """
    mask = np.ones(N_ACTIONS, dtype=bool)
    if position == 0:
        mask[3] = False
    return mask

def apply_mask_to_logits(logits: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
    # Lieto masku paketes logitiem
    mask = torch.ones(len(positions), N_ACTIONS, dtype=torch.bool, device=logits.device)
    for b, pos in enumerate(positions):
        mask[b] = torch.tensor(get_valid_mask(int(pos)), dtype=torch.bool)
    return logits.masked_fill(~mask, float("-inf"))


# 4. solis - Atlīdzības funkcija (Clean Baseline)
#
# Tīrs PnL signāls bez reward shaping:
#   - Nav HOLD soda → aģents var brīvi gaidīt
#   - Nav OPEN_PENALTY → darījuma atvēršana netiek sodīta
#   - Nav SHORT_TRADE_PEN → skalping atļauts
#   - Nav labvēlības perioda → GRACE_PERIOD_STEPS=0, tātad nav nozīmes
#   - INVALID_ACTION_PEN saglabāts → aizsargā pret loģiskām kļūdām
#   - REWARD_CLIP saglabāts → stabilizē apmācību

def compute_reward(action, position, floating_eq, prev_balance,
                   peak_eq, is_valid, steps_since_close,
                   opened_position,
                   bars_in_trade,
                   cfg=EXPERIMENT):
    """
    Clean Baseline atlīdzības politika:
    1. Nederīga darbība  = INVALID_ACTION_PEN (-0.05)
    2. HOLD bez pozīcijas = 0.0  (nav soda — atšķirībā no Basic Mode)
    3. Visi pārējie gadījumi = log_ret (tīrs PnL, bez korekcijām)
    4. Vērtības tiek apgrieztas ±REWARD_CLIP

    Šāda vienkāršota atlīdzība ļauj novērtēt, vai tirgus datos
    ir izmantojams signāls bez jebkādas reward shaping ietekmes.
    """
    if not is_valid and action != 0:
        return cfg["INVALID_ACTION_PEN"]

    # HOLD bez pozīcijas — nav soda (atšķirībā no Basic Mode, kur -0.003).
    # GRACE_PERIOD_STEPS=0, tāpēc šis nosacījums vienmēr atgriež 0.0.
    if action == 0 and position == 0:
        return 0.0

    # Tīrs log-atgriešanās — vienīgais signāls aģentam.
    log_ret = math.log(floating_eq / (prev_balance + 1e-9) + 1e-9)
    reward  = log_ret

    # Nav OPEN_PENALTY → opened_position karogs netiek izmantots.
    # Nav SHORT_TRADE_PEN → bars_in_trade netiek pārbaudīts.
    # Nav DD_COEF → drawdown netiek sodīts.

    return float(np.clip(reward, -cfg["REWARD_CLIP"], cfg["REWARD_CLIP"]))


# 5. solis - Tirdzniecības vide (TradingEnv)
# Viena tirdzniecības aģenta simulators. Savienojas ar PPO caur standarta saskarni
# reset() / step().
#
# Galvenie dizaina lēmumi:
# Lēmums tiek pieņemts pēc pašreizējās sveces stāvokļa, bet izpilde vienmēr pēc
# nākamās sveces OPEN cenas, kas novērš nākotnes datu noplūdi.
# unrealized_pnl tiek aprēķināts pēc pašreizējās sveces CLOSE (tikai novērošanai).
# steps_since_close un bars_in_trade ir reward shaping skaitītāji
# (Clean Baseline tos neizmanto, bet glabā struktūras saderības dēļ).

class TradingEnv:
    def __init__(self, X, close_prices, open_prices, timestamps,
                 start_idx, end_idx,
                 use_commission=True, use_slippage=True,
                 initial_balance=1000.0, rng=None):
        self.X               = X
        self.close_prices    = close_prices
        self.open_prices     = open_prices
        self.timestamps      = timestamps
        self.start_idx       = start_idx
        self.end_idx         = end_idx
        self.use_commission  = use_commission
        self.use_slippage    = use_slippage
        self.initial_balance = initial_balance
        self.rng             = rng
        self._last_trade     = None
        self.reset()

    def reset(self, fixed_start=None):
        """
        Atiestata vidi sākotnējā stāvoklī.
        Apmācības laikā (fixed_start=None) sākuma pozīcija tiek izvēlēta nejauši
        pieļaujamā diapazonā — tas dod daudzveidību sākuma apstākļos starp
        paralēlajām VecEnv vidēm.
        Bektesta laikā tiek izsaukts ar fixed_start=split_idx deterministiskam sākumam.
        """
        if fixed_start is not None:
            self.i = fixed_start
        else:
            max_start = max(self.start_idx, self.end_idx - SEQ_LEN - 2)
            self.i = random.randint(self.start_idx, max_start)

        self.position         = 0
        self.entry_price      = 0.0
        self.balance          = self.initial_balance
        self.peak_eq          = self.initial_balance
        self.n_trades         = 0
        self.current_trade    = None
        self._last_trade      = None
        # steps_since_close — glabāts struktūras saderības dēļ, bet Clean Baseline
        # to neizmanto (GRACE_PERIOD_STEPS=0, HOLD_PENALTY_NO_POS=0.0).
        self.steps_since_close = 999
        self.bars_in_trade     = 0    # glabāts struktūras saderības dēļ
        return self._obs()

    def _obs(self):
        """
        Veido novērojumu: SEQ_LEN sveces logu ar tirgus pazīmēm un divas aģenta
        stāvokļa kolonnas (position, unrealized_pnl).
        Šāda savienošana ļauj modelim vienlaicīgi redzēt tirgus vēsturi un savu
        pozīciju bez atsevišķa stāvokļa iegulšanas.
        """
        start  = self.i - SEQ_LEN
        window = self.X[start:self.i].copy()
        price  = float(self.close_prices[self.i])
        unreal_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unreal_pnl = float(np.clip(
                (price - self.entry_price) / self.entry_price * self.position,
                -1.0, 1.0))
        pos_col = np.full((SEQ_LEN, 1), self.position, dtype=np.float32)
        pnl_col = np.full((SEQ_LEN, 1), unreal_pnl,   dtype=np.float32)
        return np.concatenate([window, pos_col, pnl_col], axis=1).astype(np.float32)

    def step(self, action):
        """
        Viens simulācijas solis.
        Secība šāda: i+=1, darbības izpilde pēc open[i], reward aprēķins.
        Šāda secība garantē nākotnes datu noplūdes neesamību: lēmums pieņemts
        uz i-1 sveces datiem, izpildīts pēc i sveces atvēršanas cenas.
        """
        self.i += 1
        done = self.i >= self.end_idx - 1

        if done:
            if self.position != 0:
                self._execute_close(self.i - 1)
            return self._obs(), 0.0, True, {}

        prev_balance    = self.balance
        valid_mask      = get_valid_mask(self.position)
        is_valid        = bool(valid_mask[action])
        opened_position = False   # karogs: vai pozīcija tika atvērta šajā solī

        if is_valid:
            if action == 1 and self.position != 1:    # LONG
                if self.position == -1:
                    self._execute_close(self.i)
                self._execute_open(self.i, direction=+1)
                opened_position = True
            elif action == 2 and self.position != -1: # SHORT
                if self.position == 1:
                    self._execute_close(self.i)
                self._execute_open(self.i, direction=-1)
                opened_position = True
            elif action == 3 and self.position != 0:  # CLOSE
                self._execute_close(self.i)

        # Inkrementējam atbilstošo skaitītāju atkarībā no pozīcijas esamības.
        # Clean Baseline šos skaitītājus neizmanto reward aprēķinā,
        # bet tie tiek glabāti struktūras saderībai ar citiem režīmiem.
        if self.position == 0:
            self.steps_since_close += 1
        else:
            self.bars_in_trade += 1

        # Peldošais kapitāls - pozīcijas pašreizējā vērtība pēc close cenas
        # (reward aprēķinam). Izpilde pie tam vienmēr pēc open (nav nākotnes
        # datu noplūdes).
        close_price = float(self.close_prices[self.i])
        if self.position != 0 and self.entry_price > 0:
            unreal = (close_price - self.entry_price) / self.entry_price * self.position
            floating_eq = self.balance * (1 + unreal)
        else:
            floating_eq = self.balance

        self.peak_eq = max(self.peak_eq, floating_eq)

        reward = compute_reward(
            action            = action,
            position          = self.position,
            floating_eq       = floating_eq,
            prev_balance      = prev_balance,
            peak_eq           = self.peak_eq,
            is_valid          = is_valid,
            steps_since_close = self.steps_since_close,
            opened_position   = opened_position,
            bars_in_trade     = self.bars_in_trade,
        )

        return self._obs(), reward, done, {}

    def _execute_open(self, idx, direction):
        """Atver pozīciju pēc open[idx] cenas, ņemot vērā komisiju un slīdēšanu."""
        price = float(self.open_prices[idx])
        if self.use_slippage:
            slip_pct, exec_price = compute_slippage(idx, direction, self.balance, price, rng=self.rng)
        else:
            slip_pct, exec_price = 0.0, price
        comm = COMMISSION_PCT if self.use_commission else 0.0
        self.balance    *= (1 - comm - slip_pct)
        self.position    = direction
        self.entry_price = exec_price
        self.bars_in_trade = 0    # atiestata pie katras jaunas atvēršanas
        self.current_trade = {
            "direction" : "LONG" if direction == 1 else "SHORT",
            "open_time" : pd.Timestamp(self.timestamps[idx]),
            "open_price": exec_price,
            "open_idx"  : idx,
            "slip_open" : slip_pct,
            "comm_open" : comm,
        }

    def _execute_close(self, idx):
        """
        Slēdz pozīciju pēc open[idx] cenas, ņemot vērā komisiju un slīdēšanu.
        Ieraksta darījuma detaļas self._last_trade turpmākai analīzei.
        Atiestata steps_since_close uz 0.
        """
        if self.current_trade is None:
            self.position = 0; self.entry_price = 0.0; return
        price     = float(self.open_prices[idx])
        close_dir = -self.position
        if self.use_slippage:
            slip_pct, exec_price = compute_slippage(idx, close_dir, self.balance, price, rng=self.rng)
        else:
            slip_pct, exec_price = 0.0, price
        comm      = COMMISSION_PCT if self.use_commission else 0.0
        entry     = self.entry_price
        price_ret = ((exec_price - entry) / entry if self.position == 1
                     else (entry - exec_price) / entry)
        pnl_pct   = price_ret - comm - slip_pct
        pnl_usd   = self.balance * pnl_pct
        self._last_trade = {
            **self.current_trade,
            "close_time"  : pd.Timestamp(self.timestamps[idx]),
            "close_price" : exec_price,
            "duration_min": (idx - self.current_trade["open_idx"]) * 5,
            "pnl_pct"     : pnl_pct * 100,
            "pnl_usd"     : pnl_usd,
            "slip_open"   : self.current_trade["slip_open"] * 100,
            "slip_close"  : slip_pct * 100,
            "comm_total"  : (self.current_trade["comm_open"] + comm) * 100,
        }
        self.balance           = max(self.balance + pnl_usd, 0.01)
        self.n_trades         += 1
        self.position          = 0
        self.entry_price       = 0.0
        self.current_trade     = None
        self.steps_since_close = 0


# 6. solis - Vektorizētā vide (VecEnv):
# N_ENVS paralēlās vides vienā pavedienā dod sākuma pozīciju daudzveidību un
# izlīdzina gradientus apmācības laikā.

class VecEnv:
    """
    Sinhrona VecEnv (viens pavediens). N_ENVS vides ar dažādiem sākuma punktiem
    dod daudzveidīgāku pieredzi bez daudzpavedienu resursu izmaksām.
    """
    def __init__(self, X, close_prices, open_prices, timestamps,
                 start_idx, end_idx, n_envs=N_ENVS,
                 use_commission=True, use_slippage=True):
        self.n_envs = n_envs
        self.envs = [
            TradingEnv(X, close_prices, open_prices, timestamps,
                       start_idx, end_idx,
                       use_commission=use_commission,
                       use_slippage=use_slippage,
                       rng=np.random.default_rng(SEED + k))
            for k in range(n_envs)
        ]

    def reset(self):
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step(self, actions):
        obs_list, rew_list, done_list = [], [], []
        for env, a in zip(self.envs, actions):
            obs, r, done, _ = env.step(int(a))
            if done:
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(r)
            done_list.append(done)
        positions = np.array([e.position for e in self.envs], dtype=np.int32)
        return (np.stack(obs_list, axis=0).astype(np.float32),
                np.array(rew_list,  dtype=np.float32),
                np.array(done_list, dtype=np.float32),
                positions)


# 7. solis - Modeļa arhitektūra (Transformer Aktieris-Kritiķis):

print("\n" + "="*60)
print("7. SOLIS: Modelis")
print("="*60)

class AttentionBlock(nn.Module):
    """
    Viens Transformer-enkodētāja bloks: Daudzgalvu Kauzālā Pašuzmanība + FFN.
    Pre-LN shēma (LayerNorm pirms attention/FFN) stabilizē apmācību bez sasilšanas.
    Kauzālā maska nodrošina, ka svece i redz tikai sveces 0..i - imitē tiešsaistes
    lēmumu bez nākotnes skatīšanas.
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        dh = d_model // n_heads
        self.scale   = dh ** -0.5
        self.n_heads = n_heads
        self.dh      = dh
        self.qkv     = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
        self.ff1     = nn.Linear(d_model, d_model * 4)
        self.ff2     = nn.Linear(d_model * 4, d_model)
        self.ln1     = nn.LayerNorm(d_model)
        self.ln2     = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

        # Kauzālā maska kā buferis - nepiedalās atgriezeniskajā caurlaidē,
        # bet tiek saglabāta ar modeli pie torch.save().
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        )

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.dh
        res = x; x = self.ln1(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        w = F.softmax(scores, dim=-1)
        o = torch.matmul(self.drop(w), v).transpose(1, 2).contiguous().view(B, T, D)
        x = res + self.drop(self.proj(o))
        res = x; x = self.ln2(x)
        return res + self.drop(self.ff2(F.gelu(self.ff1(x))))


class TransformerActorCritic(nn.Module):
    """
    Aktieris-Kritiķis virs Transformer-enkodētāja.

    Ievade: SEQ_LEN sveces secība (INPUT_DIM pazīmes katrai).
    Lēmuma pieņemšanai tiek izmantots tikai pēdējais secības tokens (x[:, -1]) -
    tas nes visa loga uzkrāto kontekstu.

    Aktieris un Kritiķis - atsevišķi divslāņu MLP ar kopīgu enkodētāju.
    Aktieria svaru inicializācija ar mazu gain=0.01 samazina sākotnējo entropiju
    un paātrina konverģenci salīdzinājumā ar standarta inicializāciju.
    """
    def __init__(self):
        super().__init__()
        self.embed  = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos    = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.01)
        self.blocks = nn.ModuleList([AttentionBlock() for _ in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(D_MODEL)
        self.actor  = nn.Sequential(nn.Linear(D_MODEL, 64), nn.Tanh(), nn.Linear(64, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(D_MODEL, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x[:, -1])
        return self.actor(x), self.critic(x)

    def act(self, x, positions: np.ndarray):
        """Stohastiska darbība rollout vākšanai (apmācība)."""
        logits, val = self(x)
        logits = apply_mask_to_logits(logits, positions)
        dist   = torch.distributions.Categorical(logits=logits)
        act    = dist.sample()
        return act, dist.log_prob(act), val.squeeze(-1), dist.entropy()

    @torch.inference_mode()
    def predict(self, x, position: int):
        """Deterministiska darbība bektestam (argmax pēc logitiem)."""
        logits, val = self(x)
        logits = apply_mask_to_logits(logits, np.array([position]))
        return int(logits.argmax(dim=-1).item()), val

model = TransformerActorCritic().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametri: {n_params:,}  |  INPUT_DIM={INPUT_DIM}")
print(f"Arhitektūra: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}, SEQ_LEN={SEQ_LEN}")


# 8. solis - Rollout buferis + GAE
# Buferis uzkrāj (obs, action, reward, done, log_prob, value) viena rollout laikā,
# tad compute_gae_and_flatten() aprēķina priekšrocības ar GAE metodi (λ=0.95)
# un atgriež izlīdzinātas priekšrocības un atgriešanās vērtības PPO atjauninājumam.

class RolloutBuffer:
    """
    Glabā vienu rollout visām N_ENVS vidēm.
    compute_gae_and_flatten() aprēķina GAE-priekšrocības un atgriež datus
    formā (T×E, ...) mini-paketes PPO atjauninājumam.
    Priekšrocības tiek normalizētas (mean=0, std=1) apmācības stabilitātei.
    """
    def __init__(self, rollout_steps, n_envs, obs_shape):
        T, E = rollout_steps, n_envs
        self.obs      = np.zeros((T, E, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros((T, E),             dtype=np.int64)
        self.rewards  = np.zeros((T, E),             dtype=np.float32)
        self.dones    = np.zeros((T, E),             dtype=np.float32)
        self.log_probs= np.zeros((T, E),             dtype=np.float32)
        self.values   = np.zeros((T, E),             dtype=np.float32)
        self.positions= np.zeros((T, E),             dtype=np.int32)
        self.ptr      = 0
        self.T        = T
        self.E        = E

    def add(self, obs, actions, rewards, dones, log_probs, values, positions):
        t = self.ptr
        self.obs[t]       = obs
        self.actions[t]   = actions
        self.rewards[t]   = rewards
        self.dones[t]     = dones
        self.log_probs[t] = log_probs
        self.values[t]    = values
        self.positions[t] = positions
        self.ptr += 1

    def full(self):
        return self.ptr >= self.T

    def clear(self):
        self.ptr = 0

    def compute_gae_and_flatten(self, last_values: np.ndarray):
        """
        Vispārinātais Priekšrocību Novērtējums (GAE) katrai videi atsevišķi.
        Iterācija apgrieztā secībā: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        A_t = δ_t + γ·λ·A_{t+1}
        Atgriešanās = Priekšrocības + Vērtības (mērķis kritikam).
        """
        T, E = self.T, self.E
        advantages = np.zeros((T, E), dtype=np.float32)
        returns    = np.zeros((T, E), dtype=np.float32)

        for e in range(E):
            last_gae = 0.0
            next_val = last_values[e]

            for t in reversed(range(T)):
                not_done = 1.0 - self.dones[t, e]
                next_v   = next_val if t == T - 1 else self.values[t + 1, e]
                delta    = (self.rewards[t, e]
                            + GAMMA * next_v * not_done
                            - self.values[t, e])
                last_gae = delta + GAMMA * GAE_LAM * not_done * last_gae
                advantages[t, e] = last_gae
                next_val         = self.values[t, e]

            returns[:, e] = advantages[:, e] + self.values[:, e]

        obs_flat     = self.obs.reshape(T * E, *self.obs.shape[2:])
        actions_flat = self.actions.reshape(T * E)
        lp_flat      = self.log_probs.reshape(T * E)
        vals_flat    = self.values.reshape(T * E)
        adv_flat     = advantages.reshape(T * E)
        ret_flat     = returns.reshape(T * E)
        pos_flat     = self.positions.reshape(T * E)

        # Priekšrocību normalizācija - standarta pieeja PPO,
        # samazina gradienta dispersiju un paātrina konverģenci.
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        return obs_flat, actions_flat, lp_flat, vals_flat, adv_flat, ret_flat, pos_flat

# 9. solis - PPO atjauninājums:
# Standarta apgrieztais PPO (Schulman et al., 2017).
# AMP (torch.cuda.amp) ieslēdzas GPU paātrinājumam.
# Explained Variance seko vērtību funkcijas kvalitātei:
# vērtība tuvu 1.0 nozīmē labu atgriešanās aproksimāciju.

def ppo_update(model, optimizer, scaler_amp, buf_data):
    """
    Viens PPO atjauninājums: PPO_EPOCHS caurlaides pa buferi mini-paketēs MINI_BATCH.
    Zudumi:
      policy loss = -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
      value loss  = MSE(V(s), returns)
      entropy     = -E[π·log π]  (bonuss par izpēti)
      total       = policy + V_COEF·value - ENT_COEF·entropy
    """
    obs, actions, old_lp, old_vals, adv, ret, positions = buf_data
    idx = np.arange(len(obs))
    pi_losses, v_losses, entropies, clip_fracs = [], [], [], []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(obs), MINI_BATCH):
            mb = idx[start:start + MINI_BATCH]
            if len(mb) < MINI_BATCH // 2:
                continue

            s_b   = torch.tensor(obs[mb],     dtype=torch.float32, device=DEVICE)
            a_b   = torch.tensor(actions[mb], dtype=torch.long,    device=DEVICE)
            lp_b  = torch.tensor(old_lp[mb],  dtype=torch.float32, device=DEVICE)
            adv_b = torch.tensor(adv[mb],     dtype=torch.float32, device=DEVICE)
            ret_b = torch.tensor(ret[mb],     dtype=torch.float32, device=DEVICE)
            pos_b = positions[mb]

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, val      = model(s_b)
                logits_masked    = apply_mask_to_logits(logits, pos_b)
                dist             = torch.distributions.Categorical(logits=logits_masked)
                new_lp           = dist.log_prob(a_b)
                entropy          = dist.entropy().mean()
                ratio            = torch.exp(new_lp - lp_b)
                clip_r           = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pi_loss          = -torch.min(ratio * adv_b, clip_r * adv_b).mean()
                v_loss           = F.mse_loss(val.squeeze(-1), ret_b)
                loss             = pi_loss + V_COEF * v_loss - ENT_COEF * entropy

            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            with torch.no_grad():
                clipped = ((ratio - 1.0).abs() > CLIP_EPS).float().mean()
            clip_fracs.append(clipped.item())
            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.item())

    # Explained Variance: cik labi vērtību funkcija izskaidro atgriešanās dispersiju.
    # EV tuvu 1.0 - kritiks ir labi kalibrēts; EV < 0 - sliktāk par nejaušu.
    ret_t  = torch.tensor(ret,      dtype=torch.float32)
    vals_t = torch.tensor(old_vals, dtype=torch.float32)
    var_y  = ret_t.var()
    ev = float(1.0 - (ret_t - vals_t).var() / (var_y + 1e-8))

    return (np.mean(pi_losses), np.mean(v_losses),
            np.mean(entropies), np.mean(clip_fracs), ev)


# 10. solis - Apmācības cikls (PPO):
#
print("\n" + "="*60)
print("10. SOLIS: Apmācība (PPO)")
print("="*60)
print(f"Kopējie soļi: {TOTAL_STEPS:,}  |  N_ENVS: {N_ENVS}  |  ROLLOUT_STEPS: {ROLLOUT_STEPS}")
print(f"Bufera izmērs vienam atjauninājumam: {N_ENVS * ROLLOUT_STEPS:,} pārejas")
print(f"PPO epohi: {PPO_EPOCHS}  |  Mini-pakete: {MINI_BATCH}")
print(f"Režīms: {EXPERIMENT['name']}  —  visi reward shaping sodi atslēgti")
print(f"Apmācība: tīrs PnL signāls (log-atgriešanās), bez OPEN_PENALTY, SHORT_TRADE_PEN, HOLD_PENALTY")

vec_env = VecEnv(X, close_prices, open_prices, timestamps,
                 start_idx=SEQ_LEN, end_idx=split_idx,
                 n_envs=N_ENVS, use_commission=True, use_slippage=True)

obs_shape  = (SEQ_LEN, N_STATE_FEATURES)
buf        = RolloutBuffer(ROLLOUT_STEPS, N_ENVS, obs_shape)
optimizer  = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
n_updates  = TOTAL_STEPS // (N_ENVS * ROLLOUT_STEPS)
# CosineAnnealingLR pakāpeniski samazina LR no 3e-4 līdz 1e-6 visas apmācības laikā,
# tas palīdz stabilizēt politiku pēdējās iterācijās.
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_updates, eta_min=1e-6)
scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# Kontrolpunktu direktorijas nosaukums ietver VERSION un SYMBOL ērtībai.
CHECKPOINT_DIR      = f"checkpoints_{VERSION}_{SYMBOL}"
CHECKPOINT_INTERVAL = max(1, n_updates // 5)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Kontrolpunkti: {CHECKPOINT_DIR}/  (ik pēc {CHECKPOINT_INTERVAL} atjauninājumiem)")

WINDOW = 10   # logs slīdošajam vidējam reward žurnālos

state     = vec_env.reset()
positions = np.array([e.position for e in vec_env.envs], dtype=np.int32)

global_step  = 0
update_num   = 0
ep_rewards   = []
ent_history  = []
action_names = ["HOLD", "LONG", "SHORT", "CLOSE"]

import time as _time
train_start = _time.time()

"""
Žurnālu tabulas virsraksts. Kolonnas:
Step - globālais apmācības soļu skaitītājs
Upd  - PPO atjauninājuma numurs
Rew(avg10) - slīdošais vidējais reward par pēdējiem 10 atjauninājumiem
pi   - politikas zudums (jāstabilizējas tuvu 0)
v    - vērtības zudums (MSE; samazinās, kritikam kalibrējoties)
ent  - politikas entropija (↓ = aģents specializējas)
clip% - apgriezto attiecību īpatsvars (ideāli 10-30%)
ev   - kritikas explained variance (tiecas uz 1.0)
lr   - pašreizējais mācīšanās ātrums (kosinusa samazinājums)
Actions - darbību sadalījums H/L/S/C rollout laikā
"""

print(f"\n{'─'*80}")
print(f"  {'Solis':>8}  {'Upd':>4}  {'Rew(avg10)':>10}  "
      f"{'pi':>7}  {'v':>6}  {'ent':>5}  {'clip%':>6}  {'ev':>6}  "
      f"{'lr':>8}  Darbības(H/L/S/C)")
print(f"{'─'*80}")

while global_step < TOTAL_STEPS:
    buf.clear()
    rollout_rewards  = []
    rollout_actions  = np.zeros(N_ACTIONS, dtype=np.int64)

    # Rollout vākšana: N_ENVS × ROLLOUT_STEPS simulācijas soļi
    for t in range(ROLLOUT_STEPS):
        s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        with torch.inference_mode():
            actions, log_probs, values, _ = model.act(s_t, positions)

        a_np  = actions.cpu().numpy()
        lp_np = log_probs.cpu().numpy()
        v_np  = values.cpu().numpy()

        next_state, rewards, dones, next_positions = vec_env.step(a_np)
        rollout_rewards.extend(rewards.tolist())

        for a in a_np:
            rollout_actions[int(a)] += 1

        buf.add(state, a_np, rewards, dones, lp_np, v_np, positions)

        state     = next_state
        positions = next_positions
        global_step += N_ENVS

    # Pēdējās vērtības sāknēšana GAE aprēķinam
    with torch.inference_mode():
        s_last = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        _, last_vals = model(s_last)
        last_values_np = last_vals.squeeze(-1).cpu().numpy()

    buf_data = buf.compute_gae_and_flatten(last_values_np)
    pi_l, v_l, ent, clip_frac, ev = ppo_update(model, optimizer, scaler_amp, buf_data)
    scheduler.step()
    update_num += 1

    mean_rew  = float(np.mean(rollout_rewards))
    ep_rewards.append(mean_rew)
    ent_history.append(ent)
    avg_rew_w = float(np.mean(ep_rewards[-WINDOW:]))

    # Entropijas tendences bultiņa: ↓ aģents specializējas, ↑ pēta, → stabila politika
    if len(ent_history) >= 6:
        half = len(ent_history) // 2
        ent_trend = np.mean(ent_history[-half//2:]) - np.mean(ent_history[:half//2])
        ent_sym = "↓" if ent_trend < -0.005 else ("↑" if ent_trend > 0.005 else "→")
    else:
        ent_sym = "?"

    progress   = global_step / TOTAL_STEPS * 100
    total_acts = rollout_actions.sum()
    act_str    = "/".join(f"{rollout_actions[i]/total_acts*100:4.1f}%" for i in range(N_ACTIONS))
    lr_cur     = scheduler.get_last_lr()[0]

    print(f"  {global_step:>8,}  {update_num:>4d}  "
          f"{avg_rew_w:>+10.5f}  "
          f"{pi_l:>+7.4f}  {v_l:>6.4f}  {ent:>4.3f}{ent_sym}  "
          f"{clip_frac*100:>5.1f}%  {ev:>+6.3f}  "
          f"{lr_cur:>8.2e}  {act_str}")

    if update_num % CHECKPOINT_INTERVAL == 0 or global_step >= TOTAL_STEPS:
        ckpt_path = os.path.join(CHECKPOINT_DIR,
                                 f"ckpt_step{global_step:07d}.pth")
        torch.save({
            "step"       : global_step,
            "update"     : update_num,
            "model"      : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
            "ep_rewards" : ep_rewards,
        }, ckpt_path)
        print(f"  ✓ Kontrolpunkts: {ckpt_path}  [{progress:.0f}% apmācības]")

total_train_time = _time.time() - train_start
print(f"\n{'─'*80}")
print(f"  Vidējais ātrums: {TOTAL_STEPS / total_train_time:.0f} soļi/sek")
print(f"  Vidējā atlīdzība (pēdējie 10 atjauninājumi): {float(np.mean(ep_rewards[-10:])):+.5f}")
print(f"  Galīgā entropija: {ent_history[-1]:.3f}  (sākums: {ent_history[0]:.3f})")
print(f"{'─'*80}")

# Galīgā modeļa saglabāšana. Faila nosaukums ietver VERSION un SYMBOL.
MODEL_PATH = f"ppo_trader_{VERSION}_{SYMBOL}.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✓ Galīgais modelis saglabāts: {MODEL_PATH}")


# 11. solis - Bektests (4 scenāriji):
# Četras komisijas/slīdēšanas kombinācijas ļauj sadalīt kopējās izmaksas
# pa komponentiem un saprast, kas tieši ēd stratēģijas peļņu:
#   1. Komisija + slīdēšana  - reālais rezultāts
#   2. Tikai komisija      - spredas/komisijas ietekmes novērtējums
#   3. Tikai slīdēšana       - tirgus ietekmes novērtējums
#   4. Bez izmaksām        - stratēģijas teorētiskais griesti

print("\n" + "="*60)
print("11. SOLIS: Bektests (4 varianti)")
print("="*60)


def compute_sharpe_realistic(trades_df, periods_per_year=252):
    """
    Sharpe koeficients, aprēķināts pēc atsevišķu darījumu atgriešanās
    (nevis pēc kapitāla līknes svecēm). Tiek gadskārtoti caur reālo
    tirdzniecības biežumu: trades_per_year = n_trades / duration_days × 365.

    Tas ir godīgāk nekā kapitāla līknes dalīšana ar √(BARS_PER_YEAR):
    pie retas tirdzniecības sveces-bāzētais Sharpe stipri pārspīlē stratēģijas
    kvalitāti, vidējojot nulles gaidīšanas periodus kā nulles atgriešanos.
    """
    if trades_df is None or len(trades_df) < 2:
        return 0.0
    rets   = trades_df["pnl_pct"].values / 100.0
    mean_r = rets.mean()
    std_r  = rets.std()
    if std_r < 1e-9:
        return 0.0
    n_trades      = len(trades_df)
    duration_days = (trades_df["close_time"].iloc[-1] -
                     trades_df["open_time"].iloc[0]).days
    if duration_days < 1:
        trades_per_year = periods_per_year
    else:
        trades_per_year = n_trades / duration_days * 365
    return float((mean_r / std_r) * np.sqrt(trades_per_year))


def run_backtest(model, use_commission, use_slippage, label="", rng_seed=42):
    """
    Palaiž apmācīto modeli uz testa izlases deterministiskā režīmā
    (argmax pēc logitiem). Fiksēts sākums un seed nodrošina reproducējamību
    starp palaišanām.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    env = TradingEnv(X, close_prices, open_prices, timestamps,
                     start_idx=split_idx, end_idx=len(df) - 1,
                     use_commission=use_commission, use_slippage=use_slippage,
                     initial_balance=1000.0, rng=rng)

    obs = env.reset(fixed_start=split_idx)

    equity_curve   = [env.balance]
    action_counts  = {i: 0 for i in range(N_ACTIONS)}
    trades         = []
    total_steps_bt = 0

    with torch.inference_mode():
        while True:
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = model.predict(s_t, env.position)
            action_counts[action] += 1
            total_steps_bt += 1

            prev_n = env.n_trades
            obs, _, done, _ = env.step(action)

            if env.n_trades > prev_n and env._last_trade is not None:
                trades.append(env._last_trade.copy())

            price = float(close_prices[min(env.i, len(close_prices) - 1)])
            if env.position != 0 and env.entry_price > 0:
                unr = (price - env.entry_price) / env.entry_price * env.position
                equity_curve.append(env.balance * (1 + unr))
            else:
                equity_curve.append(env.balance)

            if done:
                break

    equity_arr = np.array(equity_curve, dtype=np.float32)
    final_bal  = float(env.balance)
    total_ret  = (final_bal - 1000.0) / 1000.0 * 100
    peak       = np.maximum.accumulate(equity_arr)
    max_dd     = float(((peak - equity_arr) / (peak + 1e-9) * 100).max())
    max_dd     = max(max_dd, 0.001)

    n_trades = len(trades)
    df_t     = pd.DataFrame(trades) if trades else pd.DataFrame()

    win_rate = avg_pnl = avg_dur = avg_slip = avg_comm = 0.0
    sharpe   = calmar  = profit_factor = avg_pnl_per_step = 0.0

    if n_trades > 0:
        win_rate      = float((df_t["pnl_usd"] > 0).sum() / n_trades * 100)
        avg_pnl       = float(df_t["pnl_pct"].mean())
        avg_dur       = float(df_t["duration_min"].mean())
        avg_slip      = float((df_t["slip_open"] + df_t["slip_close"]).mean()) if use_slippage else 0.0
        avg_comm      = float(df_t["comm_total"].mean()) if use_commission else 0.0
        total_pnl_usd = float(df_t["pnl_usd"].sum())
        sharpe        = compute_sharpe_realistic(df_t)
        calmar        = total_ret / max_dd
        wins_sum      = float(df_t.loc[df_t["pnl_usd"] > 0, "pnl_usd"].sum())
        losses_sum    = float(df_t.loc[df_t["pnl_usd"] < 0, "pnl_usd"].abs().sum())
        profit_factor = wins_sum / (losses_sum + 1e-9)
        avg_pnl_per_step = total_pnl_usd / (total_steps_bt + 1e-9)

    return {
        "label"           : label,
        "balance"         : final_bal,
        "total_ret"       : total_ret,
        "max_dd"          : max_dd,
        "sharpe"          : sharpe,
        "calmar"          : calmar,
        "profit_factor"   : profit_factor,
        "avg_pnl_per_step": avg_pnl_per_step,
        "n_trades"        : n_trades,
        "win_rate"        : win_rate,
        "avg_pnl_pct"     : avg_pnl,
        "avg_dur_min"     : avg_dur,
        "avg_slip_pct"    : avg_slip,
        "avg_comm_pct"    : avg_comm,
        "action_counts"   : action_counts,
        "trades"          : trades,
        "equity_curve"    : equity_arr,
    }


SCENARIOS = [
    {"label": "1. Komisija + slīdēšana          ", "comm": True,  "slip": True },
    {"label": "2. Tikai komisija              ", "comm": True,  "slip": False},
    {"label": "3. Tikai slīdēšana              ", "comm": False, "slip": True },
    {"label": "4. Bez izmaksām               ", "comm": False, "slip": False},
]

results = []
for scen in SCENARIOS:
    print(f"\n{'─'*60}  {scen['label']}")
    r = run_backtest(model, use_commission=scen["comm"],
                     use_slippage=scen["slip"], label=scen["label"], rng_seed=SEED)
    results.append(r)
    # Sharpe tiek aprēķināts pēc darījumu atgriešanās (nevis pēc kapitāla līknes svecēm).
    # Sīkāk aprakstīts compute_sharpe_realistic() funkcijā.
    print(f"  Bilance: ${r['balance']:,.2f} | Ienesīgums: {r['total_ret']:+.2f}% | "
          f"MaxDD: {r['max_dd']:.2f}% | Sharpe: {r['sharpe']:.3f} | "
          f"Calmar: {r['calmar']:.3f} | PF: {r['profit_factor']:.3f} | "
          f"Darījumi: {r['n_trades']} | Win%: {r['win_rate']:.1f}%")


# 12. solis - Rezultātu kopsavilkuma tabula:
print("\n\n" + "="*70)
print("12. SOLIS: Rezultāti")
print("="*70)
COL = 30
print(f"\n  {'Metrika':<26}  " + "  ".join(f"{s['label']:>{COL}}" for s in SCENARIOS))
print("  " + "─" * (26 + (COL + 2) * len(SCENARIOS) + 4))

def row(label, fn):
    vals = [fn(r) for r in results]
    print(f"  {label:<26}  " + "  ".join(f"{v:>{COL}}" for v in vals))

row("Galīgā bilance",          lambda r: f"${r['balance']:,.2f}")
row("Ienesīgums",              lambda r: f"{r['total_ret']:+.2f}%")
row("Maks. kritums",           lambda r: f"{r['max_dd']:.2f}%")
row("Sharpe (pēc darījumiem)", lambda r: f"{r['sharpe']:.3f}")
row("Calmar koeficients",      lambda r: f"{r['calmar']:.3f}")
row("Peļņas faktors",          lambda r: f"{r['profit_factor']:.3f}")
row("Vid. PnL / solis ($)",    lambda r: f"{r['avg_pnl_per_step']:.6f}")
row("Darījumi",                lambda r: str(r['n_trades']))
row("Uzvaru īpatsvars",        lambda r: f"{r['win_rate']:.1f}%")
row("Vid. darījuma P&L",       lambda r: f"{r['avg_pnl_pct']:+.3f}%")
row("Vid. ilgums (h)",         lambda r: f"{r['avg_dur_min']/60:.1f}")
row("Vid. slip (ieeja+izeja)", lambda r: f"{r['avg_slip_pct']:.4f}%" if r['avg_slip_pct'] > 0 else "—")
row("Vid. komisija",           lambda r: f"{r['avg_comm_pct']:.4f}%" if r['avg_comm_pct'] > 0 else "—")

# Izmaksu sadalījums: salīdzinām scenāriju bez izmaksām (4) ar pārējiem
ret_clean = results[3]["total_ret"]
print(f"\n  Zudumi no komisijām        : {ret_clean - results[1]['total_ret']:+.2f}%")
print(f"  Zudumi no slīdēšanas        : {ret_clean - results[2]['total_ret']:+.2f}%")
print(f"  Kopējie zudumi             : {ret_clean - results[0]['total_ret']:+.2f}%")

print(f"\n  Signāli (variants 1):")
ac = results[0]["action_counts"]
total_sig = max(sum(ac.values()), 1)
for aid, name in enumerate(["HOLD", "LONG", "SHORT", "CLOSE"]):
    print(f"    {name:6s}: {ac.get(aid,0):>8,}  ({ac.get(aid,0)/total_sig*100:.1f}%)")

r1 = results[0]
if r1["trades"]:
    df_trades = pd.DataFrame(r1["trades"])
    print(f"\n  Darījumi: LONG={(df_trades['direction']=='LONG').sum()}  SHORT={(df_trades['direction']=='SHORT').sum()}")

    def print_trades(title, df_t):
        print(f"\n  {'─'*58}\n  {title}")
        for rank, (_, t) in enumerate(df_t.iterrows(), 1):
            h, m = divmod(int(abs(t["duration_min"])), 60)
            sign = "+" if t["pnl_usd"] >= 0 else ""
            print(f"  {rank}. {t['direction']:5s} | "
                  f"{str(t['open_time'])[:16]} → {str(t['close_time'])[:16]} | "
                  f"{h}h {m}m | P&L: {sign}{t['pnl_usd']:.2f}$ ({sign}{t['pnl_pct']:.3f}%) | "
                  f"slip: {t['slip_open']:.3f}%/{t['slip_close']:.3f}%")

    print_trades("TOP 5 IENESĪGĀKIE", df_trades.nlargest(5, "pnl_usd"))
    print_trades("TOP 5 ZAUDĪGĀKIE",  df_trades.nsmallest(5, "pnl_usd"))

    CSV_PATH = f"backtest_{VERSION}_{SYMBOL}_trades.csv"
    df_trades.to_csv(CSV_PATH, index=False)
    print(f"\n  ✓ Darījumu žurnāls: {CSV_PATH}")


# 13. solis - Eksperimenta rezultātu saglabāšana:
# Katrs palaišanas reizē pievieno vienu rindu experiments.csv.
# Fails uzkrāj eksperimentu vēsturi turpmākai analīzei un salīdzināšanai.
import csv

def save_experiment_result(cfg, metrics, path="experiments.csv"):
    """
    Apvieno eksperimenta konfigurāciju (EXPERIMENT) un bektesta metriku.
    Virsraksts tiek pievienots automātiski pie faila pirmās izveides.
    """
    row_data = {**cfg, **metrics}
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row_data.keys())
        if write_header:
            w.writeheader()
        w.writerow(row_data)
    print(f"  ✓ Rezultāts pievienots: {path}")

main_result = results[0]
experiment_metrics = {
    "total_ret"       : round(main_result["total_ret"], 4),
    "max_dd"          : round(main_result["max_dd"], 4),
    "sharpe"          : round(main_result["sharpe"], 4),
    "calmar"          : round(main_result["calmar"], 4),
    "profit_factor"   : round(main_result["profit_factor"], 4),
    "avg_pnl_per_step": round(main_result["avg_pnl_per_step"], 8),
    "n_trades"        : main_result["n_trades"],
    "win_rate"        : round(main_result["win_rate"], 2),
    "avg_pnl_pct"     : round(main_result["avg_pnl_pct"], 4),
    "final_balance"   : round(main_result["balance"], 2),
    "D_MODEL"         : D_MODEL,
    "N_LAYERS"        : N_LAYERS,
    "SEQ_LEN"         : SEQ_LEN,
    "N_ENVS"          : N_ENVS,
    "ROLLOUT_STEPS"   : ROLLOUT_STEPS,
    "PPO_EPOCHS"      : PPO_EPOCHS,
    "TOTAL_STEPS"     : TOTAL_STEPS,
}

print(f"\n{'─'*70}")
save_experiment_result(EXPERIMENT, experiment_metrics)

print(f"\n{'='*70}\nGATAVS - versija {VERSION}  |  simbols: {SYMBOL}  |  eksperiments: {EXPERIMENT['name']}\n{'='*70}")