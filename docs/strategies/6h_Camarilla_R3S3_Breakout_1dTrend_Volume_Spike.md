# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.721 | +58.6% | -9.9% | 137 | PASS |
| ETHUSDT | 0.283 | +35.8% | -10.6% | 127 | PASS |
| SOLUSDT | 0.652 | +88.8% | -22.6% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.045 | -4.7% | -8.5% | 57 | FAIL |
| ETHUSDT | 0.431 | +12.4% | -7.5% | 45 | PASS |
| SOLUSDT | -0.821 | -7.9% | -15.3% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Daily Camarilla pivot levels (R3/S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    close_1d_prev = np.concatenate([[close_1d_prev[0]], close_1d_prev[:-1]])
    
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and 1d uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike and 1d downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend turns down
            if close[i] < S3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend turns up
            if close[i] > R3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 1d trend filter and volume spike capture strong institutional moves.
# Long when price breaks above R3 (strong resistance) with volume confirmation in 1d uptrend.
# Short when price breaks below S3 (strong support) with volume confirmation in 1d downtrend.
# R3/S3 are stronger levels than R1/S1, leading to fewer but higher-quality trades.
# Volume spike (>2x average) ensures conviction behind the breakout.
# Designed for 6h timeframe to target 12-37 trades/year, avoiding overtrading.
# Works in bull markets (breaks above R3 in uptrend) and bear markets (breaks below S3 in downtrend).
```

## Last Updated
2026-05-07 13:27
