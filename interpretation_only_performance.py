
import pandas as pd

GPT_df = pd.read_csv('')

# --- Rename columns ---
GPT_df = GPT_df.rename(columns={
    'record': 'PID',
    'step_1': 'step1_GPT',
    'step_1_pH': 'step1_pH_GPT',
    'step_2': 'step2_GPT',
    'step_2_pH': 'step2_pH_GPT',
    'step_3': 'step3_GPT',
    'step_3_pH': 'step3_pH_GPT',
    'overall_pH': 'overall_pH_GPT'

})

LLAMA_df = pd.read_csv('')

# --- Rename columns ---
LLAMA_df = LLAMA_df.rename(columns={
    'record': 'PID',
    'step_1': 'step1_LLAMA',
    'step_1_pH': 'step1_pH_LLAMA',
    'step_2': 'step2_LLAMA',
    'step_2_pH': 'step2_pH_LLAMA',
    'step_3': 'step3_LLAMA',
    'step_3_pH': 'step3_pH_LLAMA',
    'overall_pH': 'overall_pH_LLAMA'

})


GPT_df.to_csv('', index=False)
LLAMA_df.to_csv('', index=False)

GPT_df.info()
LLAMA_df.info()

# only need real pH values
import pandas as pd

realpH_df = pd.read_excel('')

# --- Rename columns ---
realpH_df = realpH_df.rename(columns={
    'File': 'PID',
    'pH': 'pH_real'
})

realpH_df.to_csv('', index=False)


import pandas as pd
from functools import reduce

realpH_path = ''
GPT_path    = ''
LLAMA_path  = ''

real_df  = pd.read_csv(realpH_path)
gpt_df   = pd.read_csv(GPT_path)
llama_df = pd.read_csv(LLAMA_path)

# Ensure key column dtype matches across all
for d in (real_df, gpt_df, llama_df):
    d['PID'] = d['PID'].astype(int)

merged_df = reduce(lambda left, right: pd.merge(left, right, on='PID', how='inner'),
                   [real_df, gpt_df, llama_df])

out_path = ''
merged_df.to_csv(out_path, index=False)

import pandas as pd

in_path  = ""
out_path = ""

df = pd.read_csv(in_path)

# Keep just PID + three columns
cols = ["PID", "pH_real", "overall_pH_GPT", "overall_pH_LLAMA"]
sub = df[cols].copy()

sub.to_csv(out_path, index=False)
print("Saved:", out_path)
print(sub.head())

"""## ANALYSIS"""

import pandas as pd
import numpy as np

path = ""
cols = ["pH_real", "overall_pH_GPT", "overall_pH_LLAMA"]
df = pd.read_csv(path)[cols].dropna()

y = df["pH_real"].to_numpy()

def simple_metrics(y_true, y_pred):
    err  = y_pred - y_true
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    r2   = 1 - np.sum(err**2) / np.sum((y_true - y_true.mean())**2)
    r    = np.corrcoef(y_true, y_pred)[0, 1]
    bias = np.mean(err)
    return mae, rmse, r2, r, bias

rows = []
for name, col in [("GPT", "overall_pH_GPT"), ("LLAMA", "overall_pH_LLAMA")]:
    p = df[col].to_numpy()
    mae, rmse, r2, r, bias = simple_metrics(y, p)
    rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": r, "MeanBias": bias})

out = pd.DataFrame(rows).round(4).sort_values("MAE")
print(out)

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load merged data
path = ""
merged_df = pd.read_csv(path, usecols=["pH_real","overall_pH_GPT","overall_pH_LLAMA"]).dropna()

def report_and_plots(pred_col, model_name):
    # Errors
    err = merged_df[pred_col] - merged_df["pH_real"]
    abs_err = np.abs(err)

    # Metrics
    mae  = abs_err.mean()
    rmse = np.sqrt((err**2).mean())
    bias = err.mean()
    r2   = r2_score(merged_df["pH_real"], merged_df[pred_col])

    # Print metrics
    print(f"\n=== {model_name} vs Real pH ===")
    print(f"{'Metric':<30} Value")
    print("-"*42)
    print(f"{'Mean Absolute Error (MAE)':<30} {mae:.4f}")
    print(f"{'Root Mean Squared Error (RMSE)':<30} {rmse:.4f}")
    print(f"{'Mean Signed Error (Bias)':<30} {bias:.4f}")
    print(f"{'R² Score':<30} {r2:.4f}")

    # Scatter: True vs Predicted
    plt.figure(figsize=(5,4))
    plt.scatter(merged_df["pH_real"], merged_df[pred_col], alpha=0.6)
    lo, hi = 6.8, 7.4  # typical pH range; adjust if needed
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="red")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True pH"); plt.ylabel("Predicted pH")
    plt.title(f"True vs Predicted pH ({model_name})")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()

# Run for GPT and LLAMA
report_and_plots("overall_pH_GPT",   "GPT")
report_and_plots("overall_pH_LLAMA", "LLAMA")

import pandas as pd, numpy as np
df = pd.read_csv(path)

# same order/unique IDs?
assert df['PID'].is_unique
df = df.sort_values('PID').reset_index(drop=True)

# how many distinct predicted values?
print(df['overall_pH_GPT'].nunique(), df['overall_pH_LLAMA'].nunique())
print(df[['pH_real','overall_pH_GPT','overall_pH_LLAMA']].describe())
print(df[['pH_real','overall_pH_GPT','overall_pH_LLAMA']].corr())


y = df['pH_real'].to_numpy()
for name, col in [('GPT','overall_pH_GPT'), ('LLAMA','overall_pH_LLAMA')]:
    p = df[col].to_numpy()
    print(name,
          "within ±0.05:", np.mean(np.abs(p-y) <= 0.05),
          "within ±0.10:", np.mean(np.abs(p-y) <= 0.10))


rmse_model_gpt   = np.sqrt(np.mean((df['overall_pH_GPT']   - y)**2))
rmse_model_llama = np.sqrt(np.mean((df['overall_pH_LLAMA'] - y)**2))
rmse_const_mean  = np.sqrt(np.mean((y.mean() - y)**2))
print(dict(rmse_model_gpt=rmse_model_gpt, rmse_model_llama=rmse_model_llama, rmse_const_mean=rmse_const_mean))

for name, col in [('GPT','overall_pH_GPT'), ('LLAMA','overall_pH_LLAMA')]:
    p = df[col].to_numpy()
    bias = np.mean(p - y)             # negative → undercall pH
    slope, intercept = np.polyfit(y, p, 1)  # slope<<1 → shrunk toward mean
    print(name, "bias:", round(bias,4), "slope:", round(slope,3), "intercept:", round(intercept,3))

def ccc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a-ma)*(b-mb))
    return (2*cov) / (va + vb + (ma-mb)**2)

print("CCC GPT:", ccc(y, df['overall_pH_GPT']))
print("CCC LLAMA:", ccc(y, df['overall_pH_LLAMA']))

import matplotlib.pyplot as plt
pred = df['overall_pH_GPT'].to_numpy()  # or LLAMA
diff = pred - y
mean = (pred + y)/2
plt.scatter(mean, diff, s=10, alpha=0.5)
plt.axhline(diff.mean(), linestyle='--')
sd = diff.std(ddof=1)
plt.axhline(diff.mean()+1.96*sd, linestyle=':')
plt.axhline(diff.mean()-1.96*sd, linestyle=':')
plt.xlabel('Mean of (Pred, True)'); plt.ylabel('Pred - True'); plt.title('Bland–Altman (GPT)'); plt.show()

import pandas as pd
import numpy as np

PATH = ""
df = pd.read_csv(PATH, usecols=["pH_real","overall_pH_GPT","overall_pH_LLAMA"]).dropna()

y = df["pH_real"].to_numpy()

def agree(y_true, y_pred, tol=0.05):
    r = np.corrcoef(y_true, y_pred)[0, 1]                 # Pearson correlation
    mae = np.mean(np.abs(y_pred - y_true))                # mean absolute error
    pct_within = np.mean(np.abs(y_pred - y_true) <= tol)  # fraction within ±tol
    return r, mae, pct_within

for name, col in [("GPT", "overall_pH_GPT"), ("LLAMA", "overall_pH_LLAMA")]:
    r, mae, pct = agree(y, df[col].to_numpy(), tol=0.05)
    print(f"{name}: r={r:.3f}  MAE={mae:.3f}  within±0.05={pct*100:.1f}%")

### BALANCE AWARE METRICS

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def fit_affine(y_true, y_pred):
    x, y = np.asarray(y_pred), np.asarray(y_true)
    varx = np.var(x)
    a = (np.cov(x, y, bias=True)[0,1] / varx) if varx > 0 else 0.0
    b = y.mean() - a * x.mean()
    return a, b

def evaluate(y_true, y_pred, label):
    err  = y_pred - y_true
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))                 # <-- manual RMSE
    bias = float(np.mean(err))
    r2   = r2_score(y_true, y_pred)
    print(f"{label:>24}  MAE={mae:.3f}  RMSE={rmse:.3f}  Bias={bias:.3f}  R2={r2:.3f}")

def calibrate_and_eval(df, pred_col, name, clip=(6.8, 7.4), test_size=0.30, seed=42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    y_tr, p_tr = train["pH_real"].to_numpy(), train[pred_col].to_numpy()
    y_te, p_te = test["pH_real"].to_numpy(),  test[pred_col].to_numpy()

    evaluate(y_te, p_te, f"{name} (raw)")

    a, b = fit_affine(y_tr, p_tr)
    p_te_cal = a * p_te + b
    if clip is not None:
        p_te_cal = np.clip(p_te_cal, *clip)

    evaluate(y_te, p_te_cal, f"{name} (affine a={a:.3f}, b={b:.3f})")
