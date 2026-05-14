# Strategy: 4h_KAMA_Trend_VolumeBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.183 | +5.2% | -20.0% | 40 | FAIL |
| ETHUSDT | 0.009 | +15.3% | -26.0% | 34 | PASS |
| SOLUSDT | 0.814 | +163.5% | -38.9% | 41 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.596 | +18.1% | -12.6% | 12 | PASS |
| SOLUSDT | 0.040 | +4.9% | -18.9% | 11 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_KAMA_Trend_VolumeBreakout
# Hypothesis: KAMA adapts to market noise, providing a smooth trend filter. 
# Combined with volume breakout above 20-period average and price closing beyond 
# KAMA ± 1.5*ATR, this captures strong momentum moves while filtering chop. 
# Works in bull/bear by following KAMA direction. Target: 20-30 trades/year.

name = "4h_KAMA_Trend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA (10,2,30) on 1d
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
        # Correct calculation of volatility (sum of absolute changes over ER period)
        vol = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                vol[i] = 0
            else:
                vol[i] = vol[i-1] + np.abs(close[i] - close[i-1])
                if i >= er_length:
                    vol[i] -= np.abs(close[i-er_length] - close[i-er_length-1]) if i-er_length-1 >= 0 else 0
        # For simplicity, use rolling sum of absolute changes
        change = np.abs(np.diff(close, prepend=close[0]))
        vol_sum = pd.Series(change).rolling(window=er_length, min_periods=1).sum().values
        # Avoid division by zero
        er = np.where(vol_sum != 0, np.abs(np.diff(close, prepend=close[0])) / vol_sum, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_1d = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # ATR (14) for dynamic threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup for KAMA and ATR
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above KAMA + 1.5*ATR with volume spike
            if close[i] > kama_1d_aligned[i] + 1.5 * atr[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below KAMA - 1.5*ATR with volume spike
            elif close[i] < kama_1d_aligned[i] - 1.5 * atr[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 05:42
