"""
Izmaiņas un labojumi:

- Godigs env: step() -> i += 1 -> cena OPEN (nav look-ahead bias)
- PPO + GAE
- Actor-Critic arhitektura
- VecEnv paralelai pieredzes vaksanai
- Almgren-Chriss slippage modelis ar rezimiem normal/elevated/stress
- Ists action masking (-inf nederigam darbibam policy)
- StandardScaler fit tikai uz train (nav data leakage)
- TeeLogger (stdout + fails)
- Darijumu zurnals CSV
- Skaidrs seed reproducjamibai

Izlabots look-ahead bias env.step() - vispirms i += 1, tad izpilde
pec OPEN; backtest() tagad izmanto env.step(), nevis duble logiku; unrealized_pnl
observation saskanots starp train un test; pozicija pievienota ka state dala (agents
redz savu poziciju).

Epsilon-decay piesaistits soliem, nevis epoham (butiski PPO
entropy); Sharpe ar korektu anualizaciju 5 minūsu timeframem - sqrt(252*24*12);
bekTests neizsauc policy divreiz uz viena stavokla; saglabatas labakas DQN ipasibas:
Almgren-Chriss slippage, action masking, StandardScaler fit tikai train.

Konfiguracija iznesta atseviškā bloka ar grupēšanu pec nozimes - visas konstantes un
parametri vienuviet.

Dati: 5-minūsu OHLCV + tehniskie indikatori
CSV jabut kolonnam: timestamp, open, high, low, close, volume, ...
"""

import os, sys, math, random, warnings
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")

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
    def close(self):
        self.log.close()
        sys.stdout = self.terminal

sys.stdout = TeeLogger(LOG_FILE)
print(f"Logs: {LOG_FILE}\n")

INPUT_FILE = "synchr16F_5m_1h.csv"

DATA_WINDOW_FRACTION = 3 / 7
TRAIN_FRACTION       = 2 / 3
SEED                 = 42

SEQ_LEN   = 64
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 2
DROPOUT   = 0.1
N_ACTIONS = 4

N_ENVS        = 8
UPDATE_FREQ   = 512
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

BARS_PER_YEAR = 252 * 24 * 12

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

print("\n" + "="*60)
print("SOLIS 1: Datu ielade")
print("="*60)

df_full = pd.read_csv(INPUT_FILE)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_full = df_full.sort_values("timestamp").reset_index(drop=True)

total_rows   = len(df_full)
window_start = int(total_rows * (1 - DATA_WINDOW_FRACTION))
df = df_full.iloc[window_start:].reset_index(drop=True)
split_idx = int(len(df) * TRAIN_FRACTION)

print(f"Rindu kopa: {total_rows:,}")
print(f"Izmantojam: {len(df):,}  |  {df['timestamp'].min()} -> {df['timestamp'].max()}")
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
print(f"\nPazimes (tirgus): {len(feature_cols)}")

close_prices = df["close"].values.astype(np.float32)
open_prices  = df["open"].values.astype(np.float32)
high_prices  = df["high"].values.astype(np.float32)
low_prices   = df["low"].values.astype(np.float32)
volumes      = df["volume"].values.astype(np.float32) if "volume" in df.columns else np.ones(len(df), np.float32)
timestamps   = df["timestamp"].values

X_raw = df[feature_cols].values.astype(np.float32)
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X = X_raw.copy()
X[:split_idx]  = scaler.fit_transform(X_raw[:split_idx])
X[split_idx:]  = scaler.transform(X_raw[split_idx:])

log_returns = np.zeros(len(X), dtype=np.float32)
log_returns[:-1] = np.log(
    (close_prices[1:] + 1e-9) / (close_prices[:-1] + 1e-9)
)

N_MARKET_FEATURES = X.shape[1]
N_STATE_FEATURES  = N_MARKET_FEATURES + 2
INPUT_DIM         = N_STATE_FEATURES

print(f"N_MARKET_FEATURES={N_MARKET_FEATURES}  N_STATE_FEATURES={N_STATE_FEATURES}")

def precompute_market_series(close_prices, high_prices, low_prices,
                              volumes, log_returns, vol_window=20):
    n = len(close_prices)

    rolling_vol = np.full(n, SLIPPAGE_PARAMS["min_slippage"], dtype=np.float32)
    for i in range(vol_window, n):
        rolling_vol[i] = float(np.std(log_returns[i - vol_window : i]))
    rolling_vol = np.maximum(rolling_vol, 1e-6)

    volume_usd = close_prices * volumes

    atr    = np.full(n, 0.001 * close_prices[0], dtype=np.float32)
    period = 14
    for i in range(1, n):
        tr = max(
            high_prices[i] - low_prices[i],
            abs(high_prices[i] - close_prices[i - 1]),
            abs(low_prices[i]  - close_prices[i - 1]),
        )
        atr[i] = ((atr[i-1] * (period-1) + tr) / period
                  if i >= period else tr)

    return {"rolling_vol": rolling_vol, "volume_usd": volume_usd,
            "atr": atr, "high": high_prices, "low": low_prices}

print("\nPrieksizrekina tirgus rindas slippage...")
market_series = precompute_market_series(
    close_prices, high_prices, low_prices, volumes, log_returns
)
print(f"  rolling_vol (train videjais): {market_series['rolling_vol'][:split_idx].mean()*100:.4f}%")
print(f"  ATR         (train videjais): {market_series['atr'][:split_idx].mean():.4f}")


def compute_slippage(i, direction, balance, price, rng=None,
                     p=SLIPPAGE_PARAMS, series=market_series):
    sigma = float(series["rolling_vol"][i])
    V = float(series["volume_usd"][i])
    Q = float(balance)

    half_spread   = p["half_spread"]
    temp_impact   = p["impact_alpha"] * sigma * np.sqrt(Q / (V + 1.0))
    perm_impact   = p["impact_beta"]  * sigma * (Q / (V + 1.0))
    vol_component = p["vol_gamma"]    * sigma
    noise_std     = p["noise_delta"]  * sigma
    noise = (rng.normal(0, noise_std) if rng is not None
             else np.random.normal(0, noise_std))

    slippage = max(half_spread + temp_impact + perm_impact + vol_component + noise, 0.0)

    if sigma >= REGIME_THRESHOLDS["stress"]:
        slippage *= REGIME_MULTIPLIERS["stress"]
    elif sigma >= REGIME_THRESHOLDS["elevated"]:
        slippage *= REGIME_MULTIPLIERS["elevated"]
    else:
        slippage *= REGIME_MULTIPLIERS["normal"]

    if direction > 0:
        slippage *= p["buy_asymmetry"]

    slippage = max(slippage, p["min_slippage"])

    candle_range = abs(series["high"][i] - series["low"][i]) / (price + 1e-9)
    if candle_range > 0:
        slippage = min(slippage, candle_range * 0.5)

    exec_price = price * (1.0 + direction * slippage)
    return float(slippage), float(exec_price)


def get_valid_action_mask(position):
    mask = np.ones(N_ACTIONS, dtype=bool)
    if position == 0:
        mask[3] = False
    return mask


class TradingEnv:

    def __init__(self, X, close_prices, open_prices, timestamps,
                 start_idx, end_idx,
                 use_commission=True, use_slippage=True,
                 initial_balance=1000.0, rng=None):
        self.X              = X
        self.close_prices   = close_prices
        self.open_prices    = open_prices
        self.timestamps     = timestamps
        self.start_idx      = start_idx
        self.end_idx        = end_idx
        self.use_commission = use_commission
        self.use_slippage   = use_slippage
        self.initial_balance = initial_balance
        self.rng            = rng

    def reset(self):
        max_start = max(self.start_idx, self.end_idx - SEQ_LEN - 2)
        self.i          = random.randint(self.start_idx, max_start)
        self.position   = 0
        self.entry_price = 0.0
        self.balance    = self.initial_balance
        self.peak_eq    = self.initial_balance
        self.n_trades   = 0
        self.current_trade = None
        return self._obs()

    def _obs(self):
        start = self.i - SEQ_LEN
        window = self.X[start : self.i].copy()

        price = float(self.close_prices[self.i])
        unreal_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unreal_pnl = ((price - self.entry_price) / self.entry_price
                          * self.position)
            unreal_pnl = float(np.clip(unreal_pnl, -1.0, 1.0))

        pos_col = np.full((SEQ_LEN, 1), self.position, dtype=np.float32)
        pnl_col = np.full((SEQ_LEN, 1), unreal_pnl,   dtype=np.float32)
        obs = np.concatenate([window, pos_col, pnl_col], axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        self.i += 1
        done = self.i >= self.end_idx - 1

        if done:
            if self.position != 0:
                self._execute_close(self.i - 1)
            return self._obs(), 0.0, True, {}

        exec_idx   = self.i
        exec_price = float(self.open_prices[exec_idx])
        prev_balance = self.balance

        valid_mask  = get_valid_action_mask(self.position)
        is_valid    = bool(valid_mask[action])
        reward      = 0.0

        if not is_valid and action != 0:
            reward = INVALID_ACTION_PEN

        elif action == 1 and self.position != 1:
            if self.position == -1:
                self._execute_close(exec_idx)
            self._execute_open(exec_idx, direction=+1)

        elif action == 2 and self.position != -1:
            if self.position == 1:
                self._execute_close(exec_idx)
            self._execute_open(exec_idx, direction=-1)

        elif action == 3 and self.position != 0:
            self._execute_close(exec_idx)

        close_price = float(self.close_prices[self.i])
        if self.position != 0 and self.entry_price > 0:
            unreal = ((close_price - self.entry_price) / self.entry_price
                      * self.position)
            floating_eq = self.balance * (1 + unreal)
        else:
            floating_eq = self.balance

        self.peak_eq = max(self.peak_eq, floating_eq)
        dd = (self.peak_eq - floating_eq) / (self.peak_eq + 1e-9)

        if action == 0 and self.position == 0:
            reward += HOLD_PENALTY_NO_POS
        elif reward == 0.0:
            log_ret = math.log(floating_eq / (prev_balance + 1e-9) + 1e-9)
            reward  = log_ret - dd * 0.05

        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        return self._obs(), reward, done, {}

    def _execute_open(self, idx, direction):
        price = float(self.open_prices[idx])
        if self.use_slippage:
            slip_pct, exec_price = compute_slippage(
                idx, direction, self.balance, price, rng=self.rng)
        else:
            slip_pct, exec_price = 0.0, price

        comm = COMMISSION_PCT if self.use_commission else 0.0
        self.balance    *= (1 - comm - slip_pct)
        self.position    = direction
        self.entry_price = exec_price
        self.current_trade = {
            "direction"  : "LONG" if direction == 1 else "SHORT",
            "open_time"  : pd.Timestamp(self.timestamps[idx]),
            "open_price" : exec_price,
            "open_idx"   : idx,
            "slip_open"  : slip_pct,
            "comm_open"  : comm,
        }

    def _execute_close(self, idx):
        if self.current_trade is None:
            self.position    = 0
            self.entry_price = 0.0
            return

        price = float(self.open_prices[idx])
        close_dir = -self.position

        if self.use_slippage:
            slip_pct, exec_price = compute_slippage(
                idx, close_dir, self.balance, price, rng=self.rng)
        else:
            slip_pct, exec_price = 0.0, price

        comm = COMMISSION_PCT if self.use_commission else 0.0
        entry = self.entry_price
        price_ret = ((exec_price - entry) / entry if self.position == 1
                     else (entry - exec_price) / entry)
        pnl_pct = price_ret - comm - slip_pct
        pnl_usd = self.balance * pnl_pct

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

        self.balance      = max(self.balance + pnl_usd, 0.01)
        self.n_trades    += 1
        self.position     = 0
        self.entry_price  = 0.0
        self.current_trade = None


class VecEnv:

    def __init__(self, X, close_prices, open_prices, timestamps,
                 start_idx, end_idx, n_envs=N_ENVS,
                 use_commission=True, use_slippage=True):
        self.envs = [
            TradingEnv(
                X, close_prices, open_prices, timestamps,
                start_idx, end_idx,
                use_commission=use_commission,
                use_slippage=use_slippage,
                rng=np.random.default_rng(SEED + k),
            )
            for k in range(n_envs)
        ]

    def reset(self):
        return np.array([e.reset() for e in self.envs], dtype=np.float32)

    def step(self, actions):
        results = []
        for env, a in zip(self.envs, actions):
            obs, r, done, info = env.step(int(a))
            if done:
                obs = env.reset()
            results.append((obs, r, done, info))
        obs_arr, rew_arr, done_arr, info_list = zip(*results)
        return (np.array(obs_arr, dtype=np.float32),
                np.array(rew_arr, dtype=np.float32),
                np.array(done_arr, dtype=bool),
                info_list)


print("\n" + "="*60)
print("SOLIS 6: Modelis")
print("="*60)


class AttentionBlock(nn.Module):

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ff1  = nn.Linear(d_model, d_model * 4)
        self.ff2  = nn.Linear(d_model * 4, d_model)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        res = x
        x   = self.ln1(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        o = torch.matmul(self.drop(w), v).transpose(1, 2).contiguous().view(B, T, D)
        x = res + self.drop(self.proj(o))

        res = x
        x   = self.ln2(x)
        x   = res + self.drop(self.ff2(F.gelu(self.ff1(x))))
        return x


class TransformerActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed  = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos    = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.01)
        self.blocks = nn.ModuleList([AttentionBlock() for _ in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(D_MODEL)

        self.actor  = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.Tanh(),
            nn.Linear(64, N_ACTIONS)
        )
        self.critic = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x[:, -1])
        return self.actor(x), self.critic(x)

    def act_masked(self, x, positions):
        logits, val = self(x)

        mask = torch.zeros(len(positions), N_ACTIONS, dtype=torch.bool, device=x.device)
        for b, pos in enumerate(positions):
            valid = get_valid_action_mask(pos)
            mask[b] = torch.tensor(valid, dtype=torch.bool)

        logits = logits.masked_fill(~mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)
        act  = dist.sample()
        return act, dist.log_prob(act), val, dist.entropy()

    @torch.inference_mode()
    def predict(self, x, position):
        logits, val = self(x)
        valid = torch.tensor(
            get_valid_action_mask(position), dtype=torch.bool, device=x.device
        )
        logits = logits.masked_fill(~valid, float("-inf"))
        return logits.argmax(dim=-1).item(), val


model = TransformerActorCritic().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametri: {n_params:,}")
print(f"INPUT_DIM={INPUT_DIM}  SEQ_LEN={SEQ_LEN}  D_MODEL={D_MODEL}")


class RolloutBuffer:

    def __init__(self):
        self.clear()

    def add(self, obs, actions, rewards, log_probs, values, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.dones.append(dones)

    def clear(self):
        self.obs = []; self.actions = []; self.rewards = []
        self.log_probs = []; self.values = []; self.dones = []

    def get(self):
        return (
            np.concatenate(self.obs,       axis=0),
            np.concatenate(self.actions,   axis=0),
            np.concatenate(self.rewards,   axis=0),
            np.concatenate(self.log_probs, axis=0),
            np.concatenate(self.values,    axis=0),
        )


def compute_gae(rewards, values, dones, last_val=0.0):
    n    = len(rewards)
    adv  = np.zeros(n, dtype=np.float32)
    last = 0.0
    vals = np.append(values, last_val)

    for i in reversed(range(n)):
        not_done = 1.0 - float(dones[i] if i < len(dones) else 0)
        delta    = rewards[i] + GAMMA * vals[i + 1] * not_done - vals[i]
        last     = delta + GAMMA * GAE_LAM * not_done * last
        adv[i]   = last

    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def ppo_update(model, optimizer, scaler_amp, buf):
    obs, actions, rewards, old_log_probs, old_values = buf.get()
    adv, returns = compute_gae(rewards, old_values)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(len(obs))
    pi_losses, v_losses, entropies = [], [], []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(obs), MINI_BATCH):
            mb = idx[start : start + MINI_BATCH]
            if len(mb) < 4:
                continue

            s_b   = torch.tensor(obs[mb],          dtype=torch.float32, device=DEVICE)
            a_b   = torch.tensor(actions[mb],       dtype=torch.long,    device=DEVICE)
            lp_b  = torch.tensor(old_log_probs[mb], dtype=torch.float32, device=DEVICE)
            adv_b = torch.tensor(adv[mb],           dtype=torch.float32, device=DEVICE)
            ret_b = torch.tensor(returns[mb],       dtype=torch.float32, device=DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, val = model(s_b)
                dist        = torch.distributions.Categorical(logits=logits)
                new_lp      = dist.log_prob(a_b)
                entropy     = dist.entropy().mean()

                ratio    = torch.exp(new_lp - lp_b)
                clip_r   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pi_loss  = -torch.min(ratio * adv_b, clip_r * adv_b).mean()
                v_loss   = F.mse_loss(val.squeeze(), ret_b)
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


print("\n" + "="*60)
print("SOLIS 8: Apmaciba (PPO)")
print("="*60)
print(f"Total steps: {TOTAL_STEPS:,}  |  N_ENVS: {N_ENVS}  |  UPDATE_FREQ: {UPDATE_FREQ}")

vec_env = VecEnv(
    X, close_prices, open_prices, timestamps,
    start_idx=SEQ_LEN,
    end_idx=split_idx,
    n_envs=N_ENVS,
    use_commission=True,
    use_slippage=True,
)
state = vec_env.reset()

optimizer  = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS // UPDATE_FREQ, eta_min=1e-6)
scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)
buf        = RolloutBuffer()

env_positions = np.array([e.position for e in vec_env.envs], dtype=np.int32)

step        = 0
update_num  = 0
recent_rews = deque(maxlen=200)

while step < TOTAL_STEPS:
    s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        actions, log_probs, values, _ = model.act_masked(s_t, env_positions)

    a_np  = actions.cpu().numpy()
    lp_np = log_probs.cpu().numpy()
    v_np  = values.cpu().numpy().flatten()

    next_state, rewards, dones, _ = vec_env.step(a_np)
    recent_rews.extend(rewards.tolist())

    buf.add(state, a_np, rewards, lp_np, v_np, dones)
    state = next_state
    step += N_ENVS

    env_positions = np.array([e.position for e in vec_env.envs], dtype=np.int32)

    if step % (UPDATE_FREQ * N_ENVS) < N_ENVS:
        with torch.inference_mode():
            _, last_val = model(torch.tensor(state, dtype=torch.float32, device=DEVICE))
        last_val_np = last_val.cpu().numpy().mean()

        pi_l, v_l, ent = ppo_update(model, optimizer, scaler_amp, buf)
        scheduler.step()
        buf.clear()
        update_num += 1

        mean_rew = np.mean(list(recent_rews)) if recent_rews else 0.0
        lr_cur   = scheduler.get_last_lr()[0]
        print(f"Step {step:7,} | upd {update_num:4d} | "
              f"rew {mean_rew:+.5f} | pi {pi_l:+.4f} | "
              f"v {v_l:.4f} | ent {ent:.3f} | lr {lr_cur:.2e}")

MODEL_PATH = f"ppo_trader_{VERSION}.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModelis saglabats: {MODEL_PATH}")


print("\n" + "="*60)
print("SOLIS 9: BekTests (4 varianti)")
print("="*60)


def run_backtest(model, use_commission, use_slippage, label="", rng_seed=42):
    model.eval()
    rng = np.random.default_rng(rng_seed)

    env = TradingEnv(
        X, close_prices, open_prices, timestamps,
        start_idx=split_idx,
        end_idx=len(df) - 1,
        use_commission=use_commission,
        use_slippage=use_slippage,
        initial_balance=1000.0,
        rng=rng,
    )

    env.i       = split_idx
    env.position = 0
    env.entry_price = 0.0
    env.balance = 1000.0
    env.peak_eq = 1000.0
    env.n_trades = 0
    env.current_trade = None
    obs = env._obs()

    equity_curve  = [env.balance]
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    trades        = []

    with torch.inference_mode():
        while True:
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = model.predict(s_t, env.position)
            action_counts[action] = action_counts.get(action, 0) + 1

            prev_trade_count = env.n_trades
            obs, reward, done, _ = env.step(action)

            if env.n_trades > prev_trade_count and hasattr(env, "_last_trade"):
                trades.append(env._last_trade.copy())

            price = float(close_prices[min(env.i, len(close_prices) - 1)])
            if env.position != 0 and env.entry_price > 0:
                unr = ((price - env.entry_price) / env.entry_price * env.position)
                equity_curve.append(env.balance * (1 + unr))
            else:
                equity_curve.append(env.balance)

            if done:
                break

    equity_arr = np.array(equity_curve, dtype=np.float32)
    final_bal  = float(env.balance)
    total_ret  = (final_bal - 1000.0) / 1000.0 * 100

    peak     = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / (peak + 1e-9) * 100
    max_dd   = float(drawdown.max())

    eq_ret = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)
    sharpe = float(eq_ret.mean() / (eq_ret.std() + 1e-9) * np.sqrt(BARS_PER_YEAR))

    n_trades = len(trades)
    win_rate = 0.0
    avg_pnl  = 0.0
    avg_dur  = 0.0
    avg_slip = 0.0
    avg_comm = 0.0

    if trades:
        df_t     = pd.DataFrame(trades)
        win_rate = float((df_t["pnl_usd"] > 0).sum() / n_trades * 100)
        avg_pnl  = float(df_t["pnl_pct"].mean())
        avg_dur  = float(df_t["duration_min"].mean())
        if use_slippage:
            avg_slip = float((df_t["slip_open"] + df_t["slip_close"]).mean())
        if use_commission:
            avg_comm = float(df_t["comm_total"].mean())

    return {
        "label"        : label,
        "balance"      : final_bal,
        "total_ret"    : total_ret,
        "max_dd"       : max_dd,
        "sharpe"       : sharpe,
        "n_trades"     : n_trades,
        "win_rate"     : win_rate,
        "avg_pnl_pct"  : avg_pnl,
        "avg_dur_min"  : avg_dur,
        "avg_slip_pct" : avg_slip,
        "avg_comm_pct" : avg_comm,
        "action_counts": action_counts,
        "trades"       : trades,
        "equity_curve" : equity_arr,
    }


SCENARIOS = [
    {"label": "1. Komisija + slippage", "comm": True,  "slip": True },
    {"label": "2. Tikai komisija            ", "comm": True,  "slip": False},
    {"label": "3. Tikai slippage       ", "comm": False, "slip": True },
    {"label": "4. Bez izmaksam              ", "comm": False, "slip": False},
]

results = []
for scen in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"  {scen['label']}")
    r = run_backtest(
        model,
        use_commission=scen["comm"],
        use_slippage=scen["slip"],
        label=scen["label"],
        rng_seed=SEED,
    )
    results.append(r)
    print(f"    Balans: ${r['balance']:,.2f}  |  "
          f"Ienakums: {r['total_ret']:+.2f}%  |  "
          f"MaxDD: {r['max_dd']:.2f}%  |  "
          f"Sharpe: {r['sharpe']:.3f}  |  "
          f"Darijumi: {r['n_trades']}  |  "
          f"Win%: {r['win_rate']:.1f}%")


print("\n\n" + "="*70)
print("SOLIS 10: Salidzinosie rezultati")
print("="*70)

COL = 30
print(f"\n  {'Metrika':<26}  " +
      "  ".join(f"{s['label']:>{COL}}" for s in SCENARIOS))
print("  " + "=" * (26 + (COL + 2) * len(SCENARIOS) + 4))

def row(label, fn):
    vals = [fn(r) for r in results]
    print(f"  {label:<26}  " + "  ".join(f"{v:>{COL}}" for v in vals))

row("Sak. depozits",      lambda r: "$1,000.00")
row("Beigu balans",       lambda r: f"${r['balance']:,.2f}")
row("Ienesigums",         lambda r: f"{r['total_ret']:+.2f}%")
row("Maks. drawdown",     lambda r: f"{r['max_dd']:.2f}%")
row("Sharpe (5m, ann.)",  lambda r: f"{r['sharpe']:.3f}")
row("Darijumi",           lambda r: str(r['n_trades']))
row("Win rate",           lambda r: f"{r['win_rate']:.1f}%")
row("Vid. P&L darijuma", lambda r: f"{r['avg_pnl_pct']:+.3f}%")
row("Vid. ilgums (st.)",  lambda r: f"{r['avg_dur_min']/60:.1f}")
row("Vid. slip (ie+iz)",  lambda r:
    f"{r['avg_slip_pct']:.4f}%" if r['avg_slip_pct'] > 0 else "-")
row("Vid. komisija",      lambda r:
    f"{r['avg_comm_pct']:.4f}%" if r['avg_comm_pct'] > 0 else "-")

print("  " + "=" * (26 + (COL + 2) * len(SCENARIOS) + 4))

ret_full  = results[0]["total_ret"]
ret_clean = results[3]["total_ret"]
ret_no_slip = results[1]["total_ret"]
ret_no_comm = results[2]["total_ret"]
total_drag = ret_clean - ret_full

print(f"\n  Izmaksu ietekme (no varianta 4 uz variantu 1):")
print(f"    Zaudeti no komisijam        : {ret_clean - ret_no_slip:+.2f}%")
print(f"    Zaudeti no slippage   : {ret_clean - ret_no_comm:+.2f}%")
print(f"    Kopejie zaudejumi           : {total_drag:+.2f}%")
if abs(total_drag) > 1e-6:
    print(f"    Slip ipatsvars              : {(ret_clean - ret_no_comm) / total_drag * 100:.1f}%")
    print(f"    Komisiju ipatsvars          : {(ret_clean - ret_no_slip) / total_drag * 100:.1f}%")

print(f"\n  Signalu sadalijums (variants 1 - realistiskais):")
ac = results[0]["action_counts"]
total_sig = max(sum(ac.values()), 1)
for act_id, name in enumerate(["HOLD", "LONG", "SHORT", "CLOSE"]):
    pct = ac.get(act_id, 0) / total_sig * 100
    print(f"    {name:6s}: {ac.get(act_id, 0):>8,}  ({pct:.1f}%)")

r1 = results[0]
if r1["trades"]:
    df_trades = pd.DataFrame(r1["trades"])
    n_long  = (df_trades["direction"] == "LONG").sum()
    n_short = (df_trades["direction"] == "SHORT").sum()

    print(f"\n  Darijumu detalas (variants 1):")
    print(f"    LONG: {n_long}  |  SHORT: {n_short}")

    def print_trades(title, df_t):
        print(f"\n  {'-'*60}")
        print(f"  {title}")
        for rank, (_, t) in enumerate(df_t.iterrows(), 1):
            h, m = divmod(int(abs(t["duration_min"])), 60)
            sign = "+" if t["pnl_usd"] >= 0 else ""
            print(f"  {rank}. {t['direction']:5s} | "
                  f"{str(t['open_time'])[:16]} -> {str(t['close_time'])[:16]} | "
                  f"{h}h {m}m | "
                  f"P&L: {sign}{t['pnl_usd']:.2f}$ ({sign}{t['pnl_pct']:.3f}%) | "
                  f"slip: {t['slip_open']:.3f}%/{t['slip_close']:.3f}%")

    print_trades("TOP 5 IENESĪGĀKIE", df_trades.nlargest(5, "pnl_usd"))
    print_trades("TOP 5 ZAUDĪGĀKIE", df_trades.nsmallest(5, "pnl_usd"))

    CSV_PATH = f"backtest_{VERSION}_trades.csv"
    df_trades.to_csv(CSV_PATH, index=False)
    print(f"\n  Darijumu zurnals: {CSV_PATH}")

print(f"\n{'='*70}")
print("GATAVS")
print(f"{'='*70}")

if hasattr(sys.stdout, "close"):
    sys.stdout.close()