# Strategy: 4h_1D_Camarilla_R1S1_Breakout_TrendVolume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.941 | +60.2% | -6.0% | 220 | PASS |
| ETHUSDT | 0.417 | +40.9% | -10.8% | 217 | PASS |
| SOLUSDT | 0.677 | +77.1% | -20.4% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.068 | -2.6% | -6.1% | 95 | FAIL |
| ETHUSDT | 1.461 | +27.3% | -7.7% | 79 | PASS |
| SOLUSDT | 0.608 | +13.9% | -8.5% | 61 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1D_Camarilla_R1S1_Breakout_TrendVolume
# Hypothesis: Breakouts from daily Camarilla R1 and S1 levels with volume confirmation and daily EMA34 trend filter.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion from extremes.
# Limits trades to 20-50 per year by requiring strict confluence of conditions.

name = "4h_1D_Camarilla_R1S1_Breakout_TrendVolume"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.1 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.1 / 4
    
    # Align daily levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below daily EMA34
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above daily EMA34
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:50
