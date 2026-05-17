"""
Izlabota look-ahead bias (galvenā izmaiņa):
Metodē step() vispirms izpildām self.i += 1, tad darījumu veicam pēc nākamās sveces open cenas
Iepriekš lēmumu pieņēmām pēc tekošās sveces close cenas ar atpakaļejošu datumu, kas radīja nereālistisku priekšrocību

Pievienoju proslīdēšanu (slippage):
Ieeju pozīcijā un izeju no tās pasliktinām par fiksētu procentu (slippage=0.0005 pēc noklusējuma) pret darījuma virzienu: pirkšana dārgāka, pārdošana lētāka

Parametrizēju komisiju un proslīdēšanu:
Vide TradingEnv un VecEnv pieņem argumentus fee un slippage, kas ļauj elastīgi pārvaldīt izmaksas apmācības un testēšanas laikā

Četri bektesta scenāriji
Pēc apmācības automātiski veicam testēšanu uz atliktās izlases četros režīmos:
    * ar komisiju un proslīdēšanu
    * bez komisijas (bet ar proslīdēšanu)
    * ar komisiju (bet bez proslīdēšanas)
    * pilnībā bez izmaksām
Rezultātus izvadām lasāmā tabulā

Precizēju equity aprēķinu novērojumam:
Tekošo cenu peldošā PnL un atlīdzības aprēķinam ņemam no close (stāvokļa informatīvumam), bet darījumu izpilde vienmēr pēc nākamā bāra open
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ielādējam datus
symm = "BTC"

df = pd.read_csv(f"{symm}.csv")
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# Pārliecināmies, ka ir kolonna 'open' - tā ir obligāta
assert "open" in df.columns, "CSV must contain 'open' column"

split    = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)

N_COLS    = train_df.shape[1]
SEQ_LEN   = 60
INPUT_DIM = N_COLS + 3
N_ACTIONS = 4   # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE

# Per-column z-score normalizācija (fit only on train)
col_mean = train_df.mean().values.astype(np.float32)
col_std  = (train_df.std().values + 1e-8).astype(np.float32)

# PPO hiperparametri
N_ENVS      = 4
UPDATE_FREQ = 256
MINI_BATCH  = 128
PPO_EPOCHS  = 4
N_STEPS     = 10_000
CLIP_EPS    = 0.2
LR          = 3e-4
GAMMA       = 0.99
GAE_LAM     = 0.95
ENT_COEF    = 0.02
V_COEF      = 0.5

# Tirdzniecības vide (izlabota!)
class TradingEnv:
    def __init__(self, df, seq_len=SEQ_LEN, fee=0.001, slippage=0.0005):
        self.df       = df
        self.seq_len  = seq_len
        self.fee      = fee
        self.slippage = slippage

    def reset(self):
        self.i        = self.seq_len
        self.position = 0
        self.entry    = 0.0
        self.equity   = 1.0
        self.peak     = 1.0
        self.n_trades = 0
        return self._obs()

    def _obs(self):
        window = self.df.iloc[self.i - self.seq_len:self.i].values.astype(np.float32)
        window = (window - col_mean) / col_std

        price = self.df.iloc[self.i]["close"]   # tekošā PnL aprēķinam izmantojam close
        pnl   = 0.0
        if self.position != 0:
            pnl = (price - self.entry) / (self.entry + 1e-8) * self.position

        extra = np.array([self.position, np.clip(pnl, -1, 1),
                          min(self.n_trades / 50.0, 1.0)], dtype=np.float32)
        return np.concatenate([window, np.tile(extra, (self.seq_len, 1))], axis=1)

    def step(self, action):
        # Izlabojums: vispirms pārvietojam laiku, tad iegūstam open cenu
        self.i += 1
        done = self.i >= len(self.df) - 1
        if done:
            # Ja dati beigušies, atgriežam tukšu stāvokli un nulles atlīdzību
            return self._obs(), 0.0, done, {}

        price_open = self.df.iloc[self.i]["open"]
        prev_eq    = self.equity
        valid      = False

        if action == 1 and self.position == 0:
            self.position = 1
            # Proslīdēšana pasliktina ieejas cenu (pērkam dārgāk)
            self.entry    = price_open * (1 + self.slippage)
            self.equity  *= (1 - self.fee)
            valid = True

        elif action == 2 and self.position == 0:
            self.position = -1
            # Proslīdēšana pasliktina ieejas cenu (pārdodam lētāk)
            self.entry    = price_open * (1 - self.slippage)
            self.equity  *= (1 - self.fee)
            valid = True

        elif action == 3 and self.position != 0:
            if self.position == 1:
                exit_price = price_open * (1 - self.slippage)   # pārdodam lētāk
            else:
                exit_price = price_open * (1 + self.slippage)   # atpērkam dārgāk
            pnl = (exit_price - self.entry) / (self.entry + 1e-8) * self.position
            self.equity *= (1 + pnl) * (1 - self.fee)
            self.position = 0
            self.n_trades += 1
            valid = True

        # Peldošais equity (atlīdzības aprēķinam)
        if self.position != 0:
            # Tekošo cenu novērojumam ņemam close (tā nepiedalās darījumos)
            price_curr = self.df.iloc[self.i]["close"]
            pnl_f = (price_curr - self.entry) / (self.entry + 1e-8) * self.position
            eq    = self.equity * (1 + pnl_f)
        else:
            eq = self.equity

        self.peak = max(self.peak, eq)
        dd        = (self.peak - eq) / (self.peak + 1e-8)

        # Atlīdzība: logaritmiskā ienesīgums ar nelielu sodu par kritumu
        reward = np.log(eq / (prev_eq + 1e-8)) - dd * 0.1
        if not valid and action != 0:
            reward -= 0.005

        return self._obs(), float(np.clip(reward, -0.1, 0.1)), done, {}

# Vektorizētā vide
class VecEnv:
    def __init__(self, df, n_envs=N_ENVS, fee=0.001, slippage=0.0005):
        self.envs = [TradingEnv(df, fee=fee, slippage=slippage) for _ in range(n_envs)]

    def reset(self):
        return np.array([e.reset() for e in self.envs])

    def step(self, actions):
        results = []
        for env, a in zip(self.envs, actions):
            obs, r, done, info = env.step(a)
            if done:
                obs = env.reset()
            results.append((obs, r, done, info))
        s, r, d, i = zip(*results)
        return np.array(s), np.array(r, dtype=np.float32), np.array(d), i

# Uzmanības bloks
class AttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
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
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        o = torch.matmul(self.drop(w), v).transpose(1, 2).contiguous().view(B, T, D)
        x = res + self.drop(self.proj(o))

        res = x;  x = self.ln2(x)
        x   = res + self.drop(self.ff2(F.gelu(self.ff1(x))))
        return x

# Aktiera-kritiķa transformera politika
class TransformerPolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, seq_len=SEQ_LEN, d_model=128, n_layers=2):
        super().__init__()
        self.embed  = nn.Linear(input_dim, d_model)
        self.pos    = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.blocks = nn.ModuleList([AttentionBlock(d_model) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.actor  = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(),
                                    nn.Linear(64, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x[:, -1])
        return self.actor(x), self.critic(x)

    def act(self, x):
        logits, val = self(x)
        dist  = torch.distributions.Categorical(logits=logits)
        act   = dist.sample()
        return act, dist.log_prob(act), val, dist.entropy()

# Pieredzes buferis
class Buffer:
    def __init__(self):
        self.clear()

    def add(self, s, a, r, lp, v):
        self.s.append(s);  self.a.append(a);  self.r.append(r)
        self.lp.append(lp);  self.v.append(v)

    def clear(self):
        self.s = [];  self.a = [];  self.r = []
        self.lp = [];  self.v = []

    def get(self):
        return (np.concatenate(self.s,  axis=0),
                np.concatenate(self.a,  axis=0),
                np.concatenate(self.r,  axis=0),
                np.concatenate(self.v,  axis=0),
                np.concatenate(self.lp, axis=0))

# GAE un PPO atjaunināšana
def compute_gae(rewards, values, last_val=0.0):
    n    = len(rewards)
    adv  = np.zeros(n, dtype=np.float32)
    last = 0.0
    vals = np.append(values, last_val)
    for i in reversed(range(n)):
        delta  = rewards[i] + GAMMA * vals[i + 1] - vals[i]
        last   = delta + GAMMA * GAE_LAM * last
        adv[i] = last
    return adv, adv + values

def ppo_update(model, opt, buf, device):
    s, a, r, v, lp = buf.get()
    adv, ret = compute_gae(r, v)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(len(s))
    pi_losses = [];  v_losses = [];  ents = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(s), MINI_BATCH):
            mb = idx[start:start + MINI_BATCH]
            if len(mb) < 4:
                continue

            sb  = torch.tensor(s[mb],   dtype=torch.float32).to(device)
            ab  = torch.tensor(a[mb],   dtype=torch.long).to(device)
            lpb = torch.tensor(lp[mb],  dtype=torch.float32).to(device)
            ab_ = torch.tensor(adv[mb], dtype=torch.float32).to(device)
            rb  = torch.tensor(ret[mb], dtype=torch.float32).to(device)

            logits, val = model(sb)
            dist    = torch.distributions.Categorical(logits=logits)
            new_lp  = dist.log_prob(ab)
            entropy = dist.entropy().mean()

            ratio   = torch.exp(new_lp - lpb)
            clip_r  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pi_loss = -torch.min(ratio * ab_, clip_r * ab_).mean()
            v_loss  = F.mse_loss(val.squeeze(), rb)
            loss    = pi_loss + V_COEF * v_loss - ENT_COEF * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())
            ents.append(entropy.item())

    return np.mean(pi_losses), np.mean(v_losses), np.mean(ents)

# Bektesta funkcija (atgriež metrikas)
def backtest(model, df, fee, slippage, initial_capital=10_000.0):
    env = TradingEnv(df, fee=fee, slippage=slippage)
    state = env.reset()
    balance = initial_capital
    position = 0
    entry = 0.0
    equity = [balance]
    trades = []

    model.eval()
    with torch.no_grad():
        while True:
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(x)
            action = torch.argmax(logits, dim=-1).item()

            # Izmantojam cenas no vides (jau ar laika nobīdi)
            price_open = df.iloc[env.i]["open"] if env.i < len(df) else df.iloc[-1]["open"]

            if action == 1 and position == 0:
                position = 1
                entry = price_open * (1 + slippage)
                balance *= (1 - fee)
            elif action == 2 and position == 0:
                position = -1
                entry = price_open * (1 - slippage)
                balance *= (1 - fee)
            elif action == 3 and position != 0:
                if position == 1:
                    exit_price = price_open * (1 - slippage)
                else:
                    exit_price = price_open * (1 + slippage)
                pnl = (exit_price - entry) / entry * position
                balance *= (1 + pnl) * (1 - fee)
                trades.append(pnl)
                position = 0

            state, _, done, _ = env.step(action)
            equity.append(balance)
            if done:
                break

    equity = np.array(equity)
    returns = np.diff(equity) / (equity[:-1] + 1e-8)
    peak = np.maximum.accumulate(equity)
    dd_pct = ((peak - equity) / (peak + 1e-8)).max() * 100
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24)

    return {
        "final_balance": balance,
        "total_return": (balance - initial_capital) / initial_capital * 100,
        "max_drawdown": dd_pct,
        "sharpe": sharpe,
        "n_trades": len(trades),
        "win_rate": (np.array(trades) > 0).mean() * 100 if trades else 0,
        "avg_pnl": np.mean(trades) * 100 if trades else 0
    }

# Apmācība
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"INPUT_DIM={INPUT_DIM}  SEQ_LEN={SEQ_LEN}  N_COLS={N_COLS}  N_ACTIONS={N_ACTIONS}")

# Apmācību veicam ar reālistiskām komisiju un proslīdēšanu
env = VecEnv(train_df, fee=0.001, slippage=0.0005)
state = env.reset()

model = TransformerPolicy().to(device)
opt = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

buf = Buffer()

for step in range(1, N_STEPS + 1):
    st = torch.tensor(state, dtype=torch.float32).to(device)
    with torch.no_grad():
        actions, logps, values, _ = model.act(st)

    a_np  = actions.cpu().numpy()
    lp_np = logps.cpu().numpy()
    v_np  = values.cpu().numpy().flatten()

    next_state, reward, done, _ = env.step(a_np)
    buf.add(state, a_np, reward, lp_np, v_np)
    state = next_state

    if step % UPDATE_FREQ == 0:
        s, a, r, v, lp = buf.get()
        mean_r = r.mean()
        pi_l, v_l, ent = ppo_update(model, opt, buf, device)
        buf.clear()
        print(f"Step {step:6d} | rew {mean_r:+.4f} | "
              f"pi {pi_l:+.4f} | v {v_l:.4f} | ent {ent:.3f} | lr {LR:.1e}")

torch.save(model.state_dict(), "rl_model_realistic.pth")
print("\nModel saved to rl_model_realistic.pth")

# Bektests 4 režīmos
print("\n" + "="*80)
print("Bektesta rezultāti (reālistiska izpilde ar open cenām)")
print("="*80)

scenarios = [
    {"fee": 0.001, "slippage": 0.0005, "label": "Ar komisiju un proslīdēšanu"},
    {"fee": 0.000, "slippage": 0.0005, "label": "Bez komisijas, ar proslīdēšanu"},
    {"fee": 0.001, "slippage": 0.0000, "label": "Ar komisiju, bez proslīdēšanas"},
    {"fee": 0.000, "slippage": 0.0000, "label": "Bez komisijas, bez proslīdēšanas"}
]

results = []
for scen in scenarios:
    res = backtest(model, test_df, fee=scen["fee"], slippage=scen["slippage"])
    results.append(res)
    print(f"\n--- {scen['label']} ---")
    print(f"  Gala bilance   : ${res['final_balance']:>10,.2f}")
    print(f"  Kopējā atdeve  : {res['total_return']:>+8.2f}%")
    print(f"  Maks. kritums  : {res['max_drawdown']:>8.2f}%")
    print(f"  Sharpe (gada)  : {res['sharpe']:>8.3f}")
    print(f"  Darījumu skaits: {res['n_trades']:>8d}")
    print(f"  Uzvaru īpatsvars: {res['win_rate']:>7.1f}%")
    print(f"  Vid. darījuma PnL: {res['avg_pnl']:>+7.3f}%")

# Kopsavilkuma tabula
print("\n" + "="*80)
print("Kopsavilkuma tabula")
print("="*80)
print(f"{'Scenārijs':<30} {'Atdeve %':>10} {'Sharpe':>8} {'Darījumi':>9} {'Uzvaras %':>9}")
for scen, res in zip(scenarios, results):
    print(f"{scen['label']:<30} {res['total_return']:>+9.2f}% {res['sharpe']:>8.3f} {res['n_trades']:>9d} {res['win_rate']:>8.1f}%")
print("="*80)