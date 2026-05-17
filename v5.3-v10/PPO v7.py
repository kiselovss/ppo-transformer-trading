"""
Izlabotas kritiskas kludas:

GAE tiek aprekins VIENREIZ, per-env, ar realiem dones
RolloutBuffer glaba dones un atgriez tos no get()
Per-env last_val: shape (N_ENVS,), nevis videja
Action masking tiek pielietots gan vaksanas, gan PPO update laika
Dati buferi tiek glabati pa-env, GAE tiek aprekins atseviski katram
Pre-allocated numpy masivi listu vieta (atrums)
fixed_start parametrs reset() ieksa manualas lauku iestatisanas vietā
VecEnv ir godigi dokumentets (sinhrons, dazadi sakumi)

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

# Logeris
VERSION  = "1.0"
LOG_FILE = f"ppo_trader_{VERSION}.log"

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

sys.stdout = TeeLogger(LOG_FILE)
print(f"Logs: {LOG_FILE}\n")

# Konfiguracija
INPUT_FILE = "synchr16F_5m_1h.csv"

DATA_WINDOW_FRACTION = 3 / 7
TRAIN_FRACTION       = 2 / 3
SEED                 = 42

SEQ_LEN   = 64
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 2
DROPOUT   = 0.1
N_ACTIONS = 4        # 0=HOLD 1=LONG 2=SHORT 3=CLOSE

# PPO
N_ENVS        = 8
ROLLOUT_STEPS = 256
MINI_BATCH    = 256
PPO_EPOCHS    = 4
TOTAL_STEPS   = 500_000
CLIP_EPS      = 0.2
LR            = 3e-4
GAMMA         = 0.99
GAE_LAM       = 0.95
ENT_COEF      = 0.02
V_COEF        = 0.5
MAX_GRAD_NORM = 0.5

COMMISSION_PCT = 0.0004

SLIPPAGE_PARAMS = {
    "half_spread"   : 0.00010,
    "impact_alpha"  : 0.10,
    "impact_beta"   : 0.01,
    "vol_gamma"     : 0.30,
    "noise_delta"   : 0.10,
    "buy_asymmetry" : 1.15,
    "min_slippage"  : 0.00005,
    "vol_window"    : 20,
}
REGIME_THRESHOLDS  = {"normal": 0.0010, "elevated": 0.0025, "stress": 0.0050}
REGIME_MULTIPLIERS = {"normal": 1.0,    "elevated": 2.0,    "stress": 3.5}

BARS_PER_YEAR = 252 * 24 * 12   # 5-minutes bari

HOLD_PENALTY_NO_POS = -0.005
INVALID_ACTION_PEN  = -0.02
REWARD_CLIP         = 0.05

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE}  |  AMP: {USE_AMP}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Solis 1 - Dati
print("\n" + "="*60)
print("Solis 1: Datu ielade")
print("="*60)

df_full = pd.read_csv(INPUT_FILE)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_full = df_full.sort_values("timestamp").reset_index(drop=True)

window_start = int(len(df_full) * (1 - DATA_WINDOW_FRACTION))
df = df_full.iloc[window_start:].reset_index(drop=True)
split_idx = int(len(df) * TRAIN_FRACTION)

print(f"Iznelojam: {len(df):,}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Train: 0..{split_idx:,}  |  Test: {split_idx:,}..{len(df):,}")

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
print(f"Pazimes (tirgus): {len(feature_cols)}")

close_prices = df["close"].values.astype(np.float32)
open_prices  = df["open"].values.astype(np.float32)
high_prices  = df["high"].values.astype(np.float32)
low_prices   = df["low"].values.astype(np.float32)
volumes      = (df["volume"].values.astype(np.float32)
                if "volume" in df.columns else np.ones(len(df), np.float32))
timestamps   = df["timestamp"].values

X_raw = df[feature_cols].values.astype(np.float32)
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X = X_raw.copy()
X[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
X[split_idx:] = scaler.transform(X_raw[split_idx:])

log_returns = np.zeros(len(X), dtype=np.float32)
log_returns[:-1] = np.log(
    (close_prices[1:] + 1e-9) / (close_prices[:-1] + 1e-9)
)

N_MARKET_FEATURES = X.shape[1]
N_STATE_FEATURES  = N_MARKET_FEATURES + 2   # + position + unrealized_pnl
INPUT_DIM         = N_STATE_FEATURES
print(f"N_STATE_FEATURES={N_STATE_FEATURES}")

# Solis 2 - Slipsana (Almgren-Chriss)
def precompute_market_series(close, high, low, volumes, log_ret, vol_window=20):
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

print("\nPrieksaprekinam slippage-rindas...")
market_series = precompute_market_series(
    close_prices, high_prices, low_prices, volumes, log_returns)

def compute_slippage(i, direction, balance, price,
                     rng=None, p=SLIPPAGE_PARAMS, series=market_series):
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

# Solis 3 - Darbibu maskesana
def get_valid_mask(position):
    """0=HOLD 1=LONG 2=SHORT 3=CLOSE. CLOSE ir aizliegts bez pozicijas."""
    mask = np.ones(N_ACTIONS, dtype=bool)
    if position == 0:
        mask[3] = False
    return mask

def apply_mask_to_logits(logits: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
    """
    Pielieto darbibu maskesanu batch logitiem.
    Izmanto GAN vaksanas, GAN PPO update laika.
    logits: (B, N_ACTIONS)
    positions: (B,) int array
    """
    mask = torch.ones(len(positions), N_ACTIONS, dtype=torch.bool, device=logits.device)
    for b, pos in enumerate(positions):
        mask[b] = torch.tensor(get_valid_mask(int(pos)), dtype=torch.bool)
    return logits.masked_fill(~mask, float("-inf"))

# Solis 4 - Tirdzniecibas vide
class TradingEnv:
    """
    Godigs env: step(), i+=1, open[i] izpildei.
    reset() pienem fixed_start - nav manualas lauku iestatisanas no arpuses.
    """

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
        Vieniga vieta stavokla inicializacijai.
        fixed_start=None - nejauss sakums (apmaciba)
        fixed_start=int - fiksets sakums (backtest)
        """
        if fixed_start is not None:
            self.i = fixed_start
        else:
            max_start = max(self.start_idx, self.end_idx - SEQ_LEN - 2)
            self.i = random.randint(self.start_idx, max_start)

        self.position      = 0
        self.entry_price   = 0.0
        self.balance       = self.initial_balance
        self.peak_eq       = self.initial_balance
        self.n_trades      = 0
        self.current_trade = None
        self._last_trade   = None
        return self._obs()

    def _obs(self):
        start  = self.i - SEQ_LEN
        window = self.X[start:self.i].copy()

        price = float(self.close_prices[self.i])
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
        i+=1 vispirms, tad open[i] izpildei.
        Atgriez (obs, reward, done, info).
        """
        self.i += 1
        done = self.i >= self.end_idx - 1

        if done:
            if self.position != 0:
                self._execute_close(self.i - 1)
            return self._obs(), 0.0, True, {}

        exec_price = float(self.open_prices[self.i])
        prev_balance = self.balance
        valid_mask   = get_valid_mask(self.position)
        is_valid     = bool(valid_mask[action])
        reward       = 0.0

        if not is_valid and action != 0:
            reward = INVALID_ACTION_PEN

        elif action == 1 and self.position != 1:    # LONG
            if self.position == -1:
                self._execute_close(self.i)
            self._execute_open(self.i, direction=+1)

        elif action == 2 and self.position != -1:   # SHORT
            if self.position == 1:
                self._execute_close(self.i)
            self._execute_open(self.i, direction=-1)

        elif action == 3 and self.position != 0:    # CLOSE
            self._execute_close(self.i)

        # Peldošais kapitals pec close (nepiedalas izpilde)
        close_price = float(self.close_prices[self.i])
        if self.position != 0 and self.entry_price > 0:
            unreal = (close_price - self.entry_price) / self.entry_price * self.position
            floating_eq = self.balance * (1 + unreal)
        else:
            floating_eq = self.balance

        self.peak_eq = max(self.peak_eq, floating_eq)
        dd = (self.peak_eq - floating_eq) / (self.peak_eq + 1e-9)

        if reward == 0.0:
            if action == 0 and self.position == 0:
                reward = HOLD_PENALTY_NO_POS
            else:
                log_ret = math.log(floating_eq / (prev_balance + 1e-9) + 1e-9)
                reward  = log_ret - dd * 0.05

        return self._obs(), float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP)), done, {}

    def _execute_open(self, idx, direction):
        price = float(self.open_prices[idx])
        if self.use_slippage:
            slip_pct, exec_price = compute_slippage(idx, direction, self.balance, price, rng=self.rng)
        else:
            slip_pct, exec_price = 0.0, price
        comm = COMMISSION_PCT if self.use_commission else 0.0
        self.balance    *= (1 - comm - slip_pct)
        self.position    = direction
        self.entry_price = exec_price
        self.current_trade = {
            "direction" : "LONG" if direction == 1 else "SHORT",
            "open_time" : pd.Timestamp(self.timestamps[idx]),
            "open_price": exec_price,
            "open_idx"  : idx,
            "slip_open" : slip_pct,
            "comm_open" : comm,
        }

    def _execute_close(self, idx):
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
        self.balance       = max(self.balance + pnl_usd, 0.01)
        self.n_trades     += 1
        self.position      = 0
        self.entry_price   = 0.0
        self.current_trade = None

# Solis 5 - VecEnv
class VecEnv:
    """
    Godiga dokumentacija: sinhrons, viens pavediens.
    N_ENVS jega - dazadi sakuma punkti, nevis CPU paralelisms.
    Katrs env sakas no nejausas vietas - agents redz visu datu kopu.
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
        obs = np.stack([e.reset() for e in self.envs], axis=0)
        return obs  # (N_ENVS, SEQ_LEN, N_STATE)

    def step(self, actions):
        """
        Atgriez obs, rewards, dones, positions pec sola.
        Pozicijas atgriezam tiesi - vajadzigas darbibu maskesanai nakama soli.
        """
        obs_list, rew_list, done_list = [], [], []
        for env, a in zip(self.envs, actions):
            obs, r, done, _ = env.step(int(a))
            if done:
                obs = env.reset()   # nejauss sakums pie reset
            obs_list.append(obs)
            rew_list.append(r)
            done_list.append(done)

        positions = np.array([e.position for e in self.envs], dtype=np.int32)
        return (np.stack(obs_list,  axis=0).astype(np.float32),
                np.array(rew_list,  dtype=np.float32),
                np.array(done_list, dtype=np.float32),
                positions)

# Solis 6 - Modelis
print("\n" + "="*60)
print("Solis 6: Modelis")
print("="*60)

class AttentionBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        dh = d_model // n_heads
        self.scale = dh ** -0.5
        self.n_heads = n_heads; self.dh = dh
        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ff1  = nn.Linear(d_model, d_model * 4)
        self.ff2  = nn.Linear(d_model * 4, d_model)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.dh
        res = x; x = self.ln1(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,H,Dh).transpose(1,2)
        k = k.view(B,T,H,Dh).transpose(1,2)
        v = v.view(B,T,H,Dh).transpose(1,2)
        w = F.softmax(torch.matmul(q, k.transpose(-2,-1)) * self.scale, dim=-1)
        o = torch.matmul(self.drop(w), v).transpose(1,2).contiguous().view(B,T,D)
        x = res + self.drop(self.proj(o))
        res = x; x = self.ln2(x)
        return res + self.drop(self.ff2(F.gelu(self.ff1(x))))


class TransformerActorCritic(nn.Module):
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
        """
        Maska tiek pielietota seit - VAKSANAS laika.
        positions: (B,) numpy array
        """
        logits, val = self(x)
        logits = apply_mask_to_logits(logits, positions)
        dist   = torch.distributions.Categorical(logits=logits)
        act    = dist.sample()
        return act, dist.log_prob(act), val.squeeze(-1), dist.entropy()

    @torch.inference_mode()
    def predict(self, x, position: int):
        """Mantkārs inference backtestam."""
        logits, val = self(x)
        logits = apply_mask_to_logits(logits, np.array([position]))
        return int(logits.argmax(dim=-1).item()), val

model = TransformerActorCritic().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametri: {n_params:,}  |  INPUT_DIM={INPUT_DIM}")

# Solis 7 - Rollout buferis + GAE
class RolloutBuffer:
    """
    Pre-allocated buferis.

    Glaba datus forma (ROLLOUT_STEPS, N_ENVS, ...) -
    katrs env iet atseviska kolonna.
    GAE tiek aprekins per-env, transitions netiek sajaukti starp envs.
    dones tiek glabati un izmantoti korekti.
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
        """obs: (E, SEQ_LEN, N_STATE), parejie: (E,)"""
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
        GAE tiek aprekins ATSEVISKI katram env.
        last_values: (N_ENVS,) - bootstrap vertiba pedeja stavokla katram env.
        Atgriez plakanus masivus mini-batch apmacibai.
        """
        T, E = self.T, self.E
        advantages = np.zeros((T, E), dtype=np.float32)
        returns    = np.zeros((T, E), dtype=np.float32)

        for e in range(E):
            last_gae = 0.0
            next_val = last_values[e]   # per-env, nevis videja

            for t in reversed(range(T)):
                # not_done nem vera realus dones
                not_done = 1.0 - self.dones[t, e]
                next_v   = next_val if t == T - 1 else self.values[t + 1, e]
                delta    = (self.rewards[t, e]
                            + GAMMA * next_v * not_done
                            - self.values[t, e])
                last_gae = delta + GAMMA * GAE_LAM * not_done * last_gae
                advantages[t, e] = last_gae
                next_val         = self.values[t, e]

            returns[:, e] = advantages[:, e] + self.values[:, e]

        # Saplacina: (T, E, ...) → (T*E, ...)
        obs_flat      = self.obs.reshape(T * E, *self.obs.shape[2:])
        actions_flat  = self.actions.reshape(T * E)
        lp_flat       = self.log_probs.reshape(T * E)
        vals_flat     = self.values.reshape(T * E)
        adv_flat      = advantages.reshape(T * E)
        ret_flat      = returns.reshape(T * E)
        pos_flat      = self.positions.reshape(T * E)

        # Advantazu normalizacija (pa visu buferi)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        return obs_flat, actions_flat, lp_flat, vals_flat, adv_flat, ret_flat, pos_flat

# Solis 8 - PPO Update
def ppo_update(model, optimizer, scaler_amp, buf_data):
    """
    darbibu maskesana tiek pielietota KATRA forward laika update laika.
    buf_data: tuple no compute_gae_and_flatten()
    """
    obs, actions, old_lp, old_vals, adv, ret, positions = buf_data
    idx = np.arange(len(obs))

    pi_losses, v_losses, entropies = [], [], []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(obs), MINI_BATCH):
            mb = idx[start:start + MINI_BATCH]
            if len(mb) < 4:
                continue

            s_b   = torch.tensor(obs[mb],     dtype=torch.float32, device=DEVICE)
            a_b   = torch.tensor(actions[mb], dtype=torch.long,    device=DEVICE)
            lp_b  = torch.tensor(old_lp[mb],  dtype=torch.float32, device=DEVICE)
            adv_b = torch.tensor(adv[mb],     dtype=torch.float32, device=DEVICE)
            ret_b = torch.tensor(ret[mb],     dtype=torch.float32, device=DEVICE)
            pos_b = positions[mb]   # numpy, priekš apply_mask_to_logits

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, val = model(s_b)

                # Ta pati maska ka vaksanas laika - ratio ir korekts
                logits_masked = apply_mask_to_logits(logits, pos_b)
                dist     = torch.distributions.Categorical(logits=logits_masked)
                new_lp   = dist.log_prob(a_b)
                entropy  = dist.entropy().mean()

                ratio    = torch.exp(new_lp - lp_b)
                clip_r   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pi_loss  = -torch.min(ratio * adv_b, clip_r * adv_b).mean()
                v_loss   = F.mse_loss(val.squeeze(-1), ret_b)
                loss     = pi_loss + V_COEF * v_loss - ENT_COEF * entropy

            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.item())

    return np.mean(pi_losses), np.mean(v_losses), np.mean(entropies)

# Solis 9 - Apmaciba
print("\n" + "="*60)
print("Solis 9: Apmaciba (PPO)")
print("="*60)
print(f"Total steps: {TOTAL_STEPS:,}  |  N_ENVS: {N_ENVS}  |  ROLLOUT_STEPS: {ROLLOUT_STEPS}")
print(f"Bufera izmers per update: {N_ENVS * ROLLOUT_STEPS:,} parrejas")

vec_env = VecEnv(X, close_prices, open_prices, timestamps,
                 start_idx=SEQ_LEN, end_idx=split_idx,
                 n_envs=N_ENVS, use_commission=True, use_slippage=True)

obs_shape  = (SEQ_LEN, N_STATE_FEATURES)
buf        = RolloutBuffer(ROLLOUT_STEPS, N_ENVS, obs_shape)
optimizer  = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
n_updates  = TOTAL_STEPS // (N_ENVS * ROLLOUT_STEPS)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_updates, eta_min=1e-6)
scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# Sakuma stavoklis
state     = vec_env.reset()                                     # (E, SEQ_LEN, N_STATE)
positions = np.array([e.position for e in vec_env.envs], dtype=np.int32)  # (E,)

global_step = 0
update_num  = 0
ep_rewards  = []

while global_step < TOTAL_STEPS:

    # -- Rollout vaksana ------------------------------------
    buf.clear()
    rollout_rewards = []

    for t in range(ROLLOUT_STEPS):
        s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)

        with torch.inference_mode():
            actions, log_probs, values, _ = model.act(s_t, positions)

        a_np  = actions.cpu().numpy()
        lp_np = log_probs.cpu().numpy()
        v_np  = values.cpu().numpy()

        next_state, rewards, dones, next_positions = vec_env.step(a_np)
        rollout_rewards.extend(rewards.tolist())

        # Glabajam positions SOSA sola (vajadzigi update laika maskesanai)
        buf.add(state, a_np, rewards, dones, lp_np, v_np, positions)

        state     = next_state
        positions = next_positions
        global_step += N_ENVS

    # -- Bootstrap pedejas vertibas - per env --------------
    with torch.inference_mode():
        s_last = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        _, last_vals = model(s_last)
        # last_values shape (N_ENVS,) - nevis videjam!
        last_values_np = last_vals.squeeze(-1).cpu().numpy()   # (E,)

    # -- GAE per-env + saplacinasana --------------------------
    buf_data = buf.compute_gae_and_flatten(last_values_np)

    # -- PPO update -----------------------------------------
    pi_l, v_l, ent = ppo_update(model, optimizer, scaler_amp, buf_data)
    scheduler.step()
    update_num += 1

    mean_rew = float(np.mean(rollout_rewards))
    ep_rewards.append(mean_rew)
    lr_cur = scheduler.get_last_lr()[0]

    print(f"Solis {global_step:7,} | upd {update_num:4d} | "
          f"rew {mean_rew:+.5f} | pi {pi_l:+.4f} | "
          f"v {v_l:.4f} | ent {ent:.3f} | lr {lr_cur:.2e}")

MODEL_PATH = f"ppo_trader_{VERSION}.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModelis saglabats: {MODEL_PATH}")

# Solis 10 - Backtests
print("\n" + "="*60)
print("Solis 10: Backtests (4 varianti)")
print("="*60)

def run_backtest(model, use_commission, use_slippage, label="", rng_seed=42):
    """
    Izmanto reset(fixed_start=split_idx) - nav manualas lauku iestatisanas.
    Backtests izsauc env.step() - neduble izpildes logiku.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    env = TradingEnv(X, close_prices, open_prices, timestamps,
                     start_idx=split_idx, end_idx=len(df) - 1,
                     use_commission=use_commission, use_slippage=use_slippage,
                     initial_balance=1000.0, rng=rng)

    # Vienigais veids inicializacijai - caur reset()
    obs = env.reset(fixed_start=split_idx)

    equity_curve  = [env.balance]
    action_counts = {i: 0 for i in range(N_ACTIONS)}
    trades        = []

    with torch.inference_mode():
        while True:
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = model.predict(s_t, env.position)
            action_counts[action] += 1

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
    eq_ret     = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)

    # Korekta anualizacija 5-minusu bariem
    sharpe = float(eq_ret.mean() / (eq_ret.std() + 1e-9) * np.sqrt(BARS_PER_YEAR))

    n_trades = len(trades)
    df_t     = pd.DataFrame(trades) if trades else pd.DataFrame()
    win_rate = float((df_t["pnl_usd"] > 0).sum() / n_trades * 100) if n_trades else 0.0
    avg_pnl  = float(df_t["pnl_pct"].mean())                        if n_trades else 0.0
    avg_dur  = float(df_t["duration_min"].mean())                    if n_trades else 0.0
    avg_slip = float((df_t["slip_open"] + df_t["slip_close"]).mean()) if (n_trades and use_slippage) else 0.0
    avg_comm = float(df_t["comm_total"].mean())                       if (n_trades and use_commission) else 0.0

    return {
        "label": label, "balance": final_bal, "total_ret": total_ret,
        "max_dd": max_dd, "sharpe": sharpe, "n_trades": n_trades,
        "win_rate": win_rate, "avg_pnl_pct": avg_pnl, "avg_dur_min": avg_dur,
        "avg_slip_pct": avg_slip, "avg_comm_pct": avg_comm,
        "action_counts": action_counts, "trades": trades, "equity_curve": equity_arr,
    }


SCENARIOS = [
    {"label": "1. Komisija + slippage", "comm": True,  "slip": True },
    {"label": "2. Tikai komisija         ", "comm": True,  "slip": False},
    {"label": "3. Tikai slippage        ", "comm": False, "slip": True },
    {"label": "4. Bez izdevumiem         ", "comm": False, "slip": False},
]

results = []
for scen in SCENARIOS:
    print(f"\n{'-'*60}  {scen['label']}")
    r = run_backtest(model, use_commission=scen["comm"],
                     use_slippage=scen["slip"], label=scen["label"], rng_seed=SEED)
    results.append(r)
    print(f"  Bilance: ${r['balance']:,.2f} | Ienakumi: {r['total_ret']:+.2f}% | "
          f"MaxDD: {r['max_dd']:.2f}% | Sharpe: {r['sharpe']:.3f} | "
          f"Darijumi: {r['n_trades']} | Win%: {r['win_rate']:.1f}%")

# -- Salidzinosā tabula ---------------------------------
print("\n\n" + "="*70)
print("Solis 11: Rezultati")
print("="*70)
COL = 30
print(f"\n  {'Metrika':<26}  " + "  ".join(f"{s['label']:>{COL}}" for s in SCENARIOS))
print("  " + "-" * (26 + (COL + 2) * len(SCENARIOS) + 4))

def row(label, fn):
    vals = [fn(r) for r in results]
    print(f"  {label:<26}  " + "  ".join(f"{v:>{COL}}" for v in vals))

row("Galiga bilance",      lambda r: f"${r['balance']:,.2f}")
row("Ienesigums",          lambda r: f"{r['total_ret']:+.2f}%")
row("Maks. kritums",       lambda r: f"{r['max_dd']:.2f}%")
row("Sharpe (5m, ann.)",   lambda r: f"{r['sharpe']:.3f}")
row("Darijumi",            lambda r: str(r['n_trades']))
row("Win rate",            lambda r: f"{r['win_rate']:.1f}%")
row("Vid. P&L darijuma",   lambda r: f"{r['avg_pnl_pct']:+.3f}%")
row("Vid. ilgums (st)",    lambda r: f"{r['avg_dur_min']/60:.1f}")
row("Vid. slip (ie+iz)",   lambda r: f"{r['avg_slip_pct']:.4f}%" if r['avg_slip_pct'] > 0 else "-")
row("Vid. komisija",       lambda r: f"{r['avg_comm_pct']:.4f}%" if r['avg_comm_pct'] > 0 else "-")

ret_clean = results[3]["total_ret"]
print(f"\n  Zudejumi no komisijam     : {ret_clean - results[1]['total_ret']:+.2f}%")
print(f"  Zudejumi no slippage    : {ret_clean - results[2]['total_ret']:+.2f}%")
print(f"  Kopējie zudejumi          : {ret_clean - results[0]['total_ret']:+.2f}%")

# Signali
print(f"\n  Signali (variants 1):")
ac = results[0]["action_counts"]
total_sig = max(sum(ac.values()), 1)
for aid, name in enumerate(["HOLD", "LONG", "SHORT", "CLOSE"]):
    print(f"    {name:6s}: {ac.get(aid,0):>8,}  ({ac.get(aid,0)/total_sig*100:.1f}%)")

# Top darijumi
r1 = results[0]
if r1["trades"]:
    df_trades = pd.DataFrame(r1["trades"])
    print(f"\n  Darijumi: LONG={( df_trades['direction']=='LONG').sum()}  SHORT={(df_trades['direction']=='SHORT').sum()}")

    def print_trades(title, df_t):
        print(f"\n  {'-'*58}\n  {title}")
        for rank, (_, t) in enumerate(df_t.iterrows(), 1):
            h, m = divmod(int(abs(t["duration_min"])), 60)
            sign = "+" if t["pnl_usd"] >= 0 else ""
            print(f"  {rank}. {t['direction']:5s} | "
                  f"{str(t['open_time'])[:16]} → {str(t['close_time'])[:16]} | "
                  f"{h}st {m}m | P&L: {sign}{t['pnl_usd']:.2f}$ ({sign}{t['pnl_pct']:.3f}%) | "
                  f"slip: {t['slip_open']:.3f}%/{t['slip_close']:.3f}%")

    print_trades("TOP 5 IENESĪGĀKIE", df_trades.nlargest(5, "pnl_usd"))
    print_trades("TOP 5 ZAUDĪGĀKIE", df_trades.nsmallest(5, "pnl_usd"))

    CSV_PATH = f"backtest_{VERSION}_trades.csv"
    df_trades.to_csv(CSV_PATH, index=False)
    print(f"\n  Darijumu zurnals: {CSV_PATH}")

print(f"\n{'='*70}\nGATAVS - versija {VERSION}\n{'='*70}")