# Strategy: 4h_Price_Above_1D_EMA200_With_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.359 | +4.5% | -16.1% | 80 | FAIL |
| ETHUSDT | 0.017 | +19.5% | -16.4% | 68 | PASS |
| SOLUSDT | 0.847 | +116.6% | -22.6% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.099 | +6.8% | -11.8% | 28 | PASS |
| SOLUSDT | 0.134 | +7.4% | -11.8% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Price_Above_1D_EMA200_With_Volume_Spike
# Hypothesis: In strong trends, price stays above/below the 200-day EMA. 
# Go long when price crosses above 1D EMA200 with volume spike and bullish 4H momentum.
# Go short when price crosses below 1D EMA200 with volume spike and bearish 4H momentum.
# The 200-day EMA acts as a strong dynamic support/resistance, filtering noise.
# Volume spike confirms institutional participation. Works in bull/bear by following the higher timeframe trend.

name = "4h_Price_Above_1D_EMA200_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA200 on 1d close
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4H momentum: EMA21 > EMA50 for bullish, EMA21 < EMA50 for bearish
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema21[i]) or 
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above 1D EMA200 + bullish 4H momentum + volume spike
            if close[i] > ema200_1d_aligned[i] and ema21[i] > ema50[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below 1D EMA200 + bearish 4H momentum + volume spike
            elif close[i] < ema200_1d_aligned[i] and ema21[i] < ema50[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below 1D EMA200 or momentum turns bearish
            if close[i] < ema200_1d_aligned[i] or ema21[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above 1D EMA200 or momentum turns bullish
            if close[i] > ema200_1d_aligned[i] or ema21[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 03:44
