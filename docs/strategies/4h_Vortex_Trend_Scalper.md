# Strategy: 4h_Vortex_Trend_Scalper

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.533 | +4.8% | -7.8% | 153 | FAIL |
| ETHUSDT | 0.142 | +26.3% | -9.9% | 149 | PASS |
| SOLUSDT | 0.933 | +95.9% | -12.2% | 105 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.064 | +6.5% | -6.5% | 51 | PASS |
| SOLUSDT | -0.862 | -3.3% | -9.4% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Vortex_Trend_Scalper
Hypothesis: Use Vortex Indicator (VI) crossovers on 4h with daily trend filter and volume confirmation to capture intermediate-term trends. Works in both bull and bear markets by filtering trades with higher timeframe trend and requiring volume confirmation, reducing false signals during choppy periods. Target: 20-40 trades/year.
"""

name = "4h_Vortex_Trend_Scalper"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # Vortex Indicator on 4h (period=14)
    tr = np.maximum(np.abs(high[1:] - low[:-1]), 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align length
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan

    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values

    vi_plus = sum_vm_plus14 / sum_tr14
    vi_minus = sum_vm_minus14 / sum_tr14

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        vi_p = vi_plus[i]
        vi_m = vi_minus[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- + daily uptrend + volume confirmation
            if vi_p > vi_m and vi_plus[i-1] <= vi_minus[i-1] and close[i] > ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ + daily downtrend + volume confirmation
            elif vi_m > vi_p and vi_minus[i-1] <= vi_plus[i-1] and close[i] < ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ or close below daily EMA34
            if vi_m > vi_p or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- or close above daily EMA34
            if vi_p > vi_m or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 21:53
