# Strategy: 4H_CAMARILLA_R1_S1_BREAKOUT_12H_EMA50_TREND_VOLUME

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.301 | +32.2% | -9.4% | 345 | PASS |
| ETHUSDT | 0.323 | +35.0% | -9.7% | 304 | PASS |
| SOLUSDT | 0.089 | +23.1% | -18.8% | 256 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.372 | -4.7% | -5.4% | 130 | FAIL |
| ETHUSDT | 0.918 | +17.8% | -9.3% | 110 | PASS |
| SOLUSDT | 0.635 | +13.9% | -6.9% | 93 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12H_EMA50_TREND_VOLUME
# Hypothesis: Camarilla R1/S1 levels on 12h chart represent strong breakout points with trend and volume confirmation.
# Price breaking above R1 with volume and 12h uptrend signals continuation long.
# Price breaking below S1 with volume and 12h downtrend signals continuation short.
# Works in bull (buy breakouts) and bear (sell breakdowns) markets by following trend.
# Target: 20-50 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12H_EMA50_TREND_VOLUME"
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
    
    # 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R1 and S1 levels from previous 12h bar (requires previous bar's data)
    camarilla_r1 = np.full(len(close_12h), np.nan)
    camarilla_s1 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        # Previous 12h bar's values
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        range_val = ph - pl
        
        # Camarilla R1 and S1 levels
        camarilla_r1[i] = pc + range_val * 1.1 / 6
        camarilla_s1[i] = pc - range_val * 1.1 / 6
    
    # EMA50 for 12h trend filter
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 4h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all 12h data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to ensure previous bar data exists
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike in uptrend
            if (high[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike in downtrend
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend reversal
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend reversal
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:16
