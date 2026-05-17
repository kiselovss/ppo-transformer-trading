import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ielādējam datus
df = pd.read_csv("BTC.csv")
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

split    = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)

N_COLS    = train_df.shape[1]
SEQ_LEN   = 60       # īsāks logs - ātrāks mācīšanās signāls
INPUT_DIM = N_COLS + 3

# normalizējam katru kolonnu ar slīdošo statistiku no treniņa kopas
col_mean = train_df.mean().values.astype(np.float32)
col_std  = train_df.std().values.astype(np.float32) + 1e-8

# ppo regulējamie parametri
N_ENVS      = 4
UPDATE_FREQ = 256
MINI_BATCH  = 128
PPO_EPOCHS  = 4
N_STEPS     = 8000
CLIP_EPS    = 0.2
LR          = 1e-3


# vide
class TradingEnv:
    """
    Darbības:
      0 = HOLD
      1 = LONG
      2 = SHORT
      3 = CLOSE
    """
    def __init__(self, df, seq_len=SEQ_LEN, fee=0.001):
        self.df      = df
        self.seq_len = seq_len
        self.fee     = fee
        self.n_cols  = df.shape[1]

    def reset(self):
        self.i        = self.seq_len
        self.position = 0     # -1, 0, 1
        self.entry    = 0.0
        self.equity   = 1.0
        self.peak     = 1.0
        self.n_trades = 0
        return self._obs()

    def _norm(self, window):
        return (window - col_mean) / col_std

    def _obs(self):
        window = self.df.iloc[self.i - self.seq_len:self.i].values.astype(np.float32)
        window = self._norm(window)                       # normalizējam!

        price = self.df.iloc[self.i]["close"]
        pnl   = 0.0
        if self.position != 0:
            pnl = (price - self.entry) / (self.entry + 1e-8) * self.position

        extra       = np.array([self.position, pnl, float(self.n_trades) / 100.0],
                               dtype=np.float32)
        extra_tiled = np.tile(extra, (self.seq_len, 1))
        return np.concatenate([window, extra_tiled], axis=1)  # (seq_len, INPUT_DIM)

    def step(self, action):
        price   = self.df.iloc[self.i]["close"]
        prev_eq = self.equity
        valid   = False
        reward  = 0.0

        #  izpildām darbību
        if action == 1 and self.position == 0:       # atveram garo pozīciju
            self.position = 1
            self.entry    = price
            self.equity  *= (1 - self.fee)
            valid = True

        elif action == 2 and self.position == 0:     # atveram īso pozīciju
            self.position = -1
            self.entry    = price
            self.equity  *= (1 - self.fee)
            valid = True

        elif action == 3 and self.position != 0:     # aizveram
            pnl = (price - self.entry) / (self.entry + 1e-8) * self.position
            self.equity *= (1 + pnl) * (1 - self.fee)
            self.position = 0
            self.n_trades += 1
            valid = True

        # sods par nederīgu darbību (mazs)
        if not valid and action != 0:
            reward -= 0.002

        # ---- mainīgais kapitāls atlīdzībai ----
        if self.position != 0:
            pnl = (price - self.entry) / (self.entry + 1e-8) * self.position
            eq  = self.equity * (1 + pnl)
        else:
            eq = self.equity

        # izņemšanas sods
        self.peak = max(self.peak, eq)
        dd        = (self.peak - eq) / (self.peak + 1e-8)

        # atlīdzība = soļa peļņa/zaudējumi - izņemšanas sods
        step_pnl = eq - prev_eq
        reward  += step_pnl * 100.0 - dd * 0.05   # mērogojam, lai gradienti nebūtu triviāli

        self.i += 1
        done = self.i >= len(self.df) - 1
        return self._obs(), float(reward), done, {}


# vektora vide
class VecEnv:
    def __init__(self, df, n_envs=N_ENVS):
        self.envs = [TradingEnv(df) for _ in range(n_envs)]

    def reset(self):
        return np.array([e.reset() for e in self.envs])

    def step(self, actions):
        out = [e.step(a) for e, a in zip(self.envs, actions)]
        s, r, d, i = zip(*out)
        obs  = []
        for idx, (env, done) in enumerate(zip(self.envs, d)):
            if done:
                obs.append(env.reset())
            else:
                obs.append(s[idx])
        return np.array(obs), np.array(r, dtype=np.float32), np.array(d), i


# uzmanības bloks
class AttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, d_model * 3, bias=False)
        self.out     = nn.Linear(d_model, d_model)
        self.ff1     = nn.Linear(d_model, d_model * 2)
        self.ff2     = nn.Linear(d_model * 2, d_model)
        self.ln1     = nn.LayerNorm(d_model)
        self.ln2     = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        res = x;  x = self.ln1(x)
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(self.drop(attn), v)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        x    = res + self.drop(self.out(out))

        res = x;  x = self.ln2(x)
        x   = res + self.drop(self.ff2(F.gelu(self.ff1(x))))
        return x


# mācīšanas politika
N_ACTIONS = 4   # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE

class TransformerPolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, seq_len=SEQ_LEN, d_model=128, n_layers=2):
        super().__init__()
        self.embed  = nn.Linear(input_dim, d_model)
        self.pos    = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, n_heads=4) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)

        # atsevišķas aktiera / kritiķa galvas
        self.actor  = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS)
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.embed(x) + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x[:, -1])
        return self.actor(x), self.critic(x)

    def act(self, x):
        logits, value = self(x)
        dist    = torch.distributions.Categorical(logits=logits)
        action  = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# buferis
class Buffer:
    def __init__(self):
        self.clear()

    def add(self, s, a, r, logp, v):
        self.s.append(s);    self.a.append(a)
        self.r.append(r);    self.logp.append(logp)
        self.v.append(v)

    def clear(self):
        self.s = [];  self.a = [];  self.r = []
        self.logp = [];  self.v = []

    def get(self):
        return (np.concatenate(self.s,    axis=0),
                np.concatenate(self.a,    axis=0),
                np.concatenate(self.r,    axis=0),
                np.concatenate(self.v,    axis=0),
                np.concatenate(self.logp, axis=0))


# vispārinātais priekšrocību novērtējums
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv  = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    vals = np.append(values, 0.0)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * vals[i + 1] - vals[i]
        last  = delta + gamma * lam * last
        adv[i] = last
    returns = adv + values
    return adv, returns


# ppo atjaunināšana (mini-pakotnēs)
def ppo_update(model, opt, buf, device):
    s_np, a_np, r_np, v_np, lp_np = buf.get()

    adv_np, ret_np = compute_gae(r_np, v_np)
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

    n = len(s_np)
    idx = np.arange(n)
    pi_losses = [];  v_losses = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, MINI_BATCH):
            mb = idx[start:start + MINI_BATCH]
            if len(mb) < 4:
                continue

            sb  = torch.tensor(s_np[mb],   dtype=torch.float32).to(device)
            ab  = torch.tensor(a_np[mb],   dtype=torch.long).to(device)
            lpb = torch.tensor(lp_np[mb],  dtype=torch.float32).to(device)
            ab_ = torch.tensor(adv_np[mb], dtype=torch.float32).to(device)
            rb  = torch.tensor(ret_np[mb], dtype=torch.float32).to(device)

            logits, val = model(sb)
            dist    = torch.distributions.Categorical(logits=logits)
            new_lp  = dist.log_prob(ab)
            entropy = dist.entropy().mean()

            ratio   = torch.exp(new_lp - lpb)
            clip_r  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pi_loss = -torch.min(ratio * ab_, clip_r * ab_).mean()
            v_loss  = F.mse_loss(val.squeeze(), rb)
            loss    = pi_loss + 0.5 * v_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())

    return np.mean(pi_losses), np.mean(v_losses)


# apmācām
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"INPUT_DIM={INPUT_DIM}  SEQ_LEN={SEQ_LEN}  N_COLS={N_COLS}  N_ACTIONS={N_ACTIONS}")

env   = VecEnv(train_df, n_envs=N_ENVS)
state = env.reset()

model = TransformerPolicy().to(device)
opt   = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS // UPDATE_FREQ)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

buf = Buffer()

for step in range(1, N_STEPS + 1):

    state_t = torch.tensor(state, dtype=torch.float32).to(device)

    with torch.no_grad():
        actions, logps, values = model.act(state_t)

    actions_np = actions.cpu().numpy()
    logps_np   = logps.cpu().numpy()
    values_np  = values.cpu().numpy().flatten()

    next_state, reward, done, _ = env.step(actions_np)

    buf.add(state, actions_np, reward, logps_np, values_np)
    state = next_state

    if step % UPDATE_FREQ == 0:
        s_np, _, r_np, _, _ = buf.get()
        mean_r = r_np.mean()

        pi_l, v_l = ppo_update(model, opt, buf, device)
        sched.step()
        buf.clear()

        print(f"Step {step:5d} | reward {mean_r:+.5f} | "
              f"pi {pi_l:+.4f} | v {v_l:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e}")


# atpakaļpārbaude
model.eval()
env_test = TradingEnv(test_df)
state    = env_test.reset()

balance  = 10_000.0
position = 0
entry    = 0.0
equity   = [balance]
trades   = []

with torch.no_grad():
    while True:
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x)
        action    = torch.argmax(logits, dim=-1).item()

        price = test_df.iloc[env_test.i]["close"]

        if action == 1 and position == 0:
            position = 1;   entry = price
            balance *= (1 - 0.001)

        elif action == 2 and position == 0:
            position = -1;  entry = price
            balance *= (1 - 0.001)

        elif action == 3 and position == 1:
            pnl      = (price - entry) / entry
            balance *= (1 + pnl) * (1 - 0.001)
            trades.append(pnl)
            position = 0

        elif action == 3 and position == -1:
            pnl      = (entry - price) / entry
            balance *= (1 + pnl) * (1 - 0.001)
            trades.append(pnl)
            position = 0

        state, _, done, _ = env_test.step(action)
        equity.append(balance)
        if done:
            break

equity = np.array(equity)
returns = np.diff(equity) / equity[:-1]
sharpe  = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 24)  # stundu → gada izteiksmē
max_dd  = ((equity.cummax() - equity) / equity.cummax()).max() * 100

print(f"\n{'='*45}")
print(f"  beigu bilance  : ${balance:>10,.2f}")
print(f"  kopējā atdeve  : {((balance-10_000)/10_000)*100:>+8.2f}%")
print(f"  maks. izņemšana: {max_dd:>8.2f}%")
print(f"  šarpa rādītājs (gada): {sharpe:>8.3f}")
print(f"  darījumi       : {len(trades):>8d}")
if trades:
    arr = np.array(trades)
    print(f"  uzvaru īpatsvars: {(arr>0).mean()*100:>7.1f}%")
    print(f"  vid. darījuma pnl: {arr.mean()*100:>+7.3f}%")
print(f"{'='*45}")