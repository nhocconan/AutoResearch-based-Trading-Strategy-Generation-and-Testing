# Strategy: 6H_CAMARILLA_R4_S4_BREAKOUT_1D_TREND_VOLUME

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.584 | +40.9% | -4.7% | 206 | PASS |
| ETHUSDT | 0.350 | +34.8% | -7.4% | 188 | PASS |
| SOLUSDT | 0.585 | +64.9% | -13.0% | 151 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.249 | -2.8% | -5.4% | 82 | FAIL |
| ETHUSDT | 1.511 | +24.8% | -5.3% | 69 | PASS |
| SOLUSDT | 0.213 | +8.2% | -4.8% | 59 | PASS |

## Code
```python
# 6H_CAMARILLA_R4_S4_BREAKOUT_1D_TREND_VOLUME
# Hypothesis: Camarilla R4/S4 levels on 1d chart represent strong breakout points.
# Price breaking above R4 with volume and 1d uptrend signals continuation long.
# Price breaking below S4 with volume and 1d downtrend signals continuation short.
# Works in bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 20-50 trades/year on 6h timeframe.

name = "6H_CAMARILLA_R4_S4_BREAKOUT_1D_TREND_VOLUME"
timeframe = "6h"
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
    
    # Daily data for Camarilla calculation and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous day (requires previous day's data)
    # Calculate for each day using previous day's H, L, C
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        # Camarilla R4 and S4 levels
        camarilla_r4[i] = pc + range_val * 1.1 / 2
        camarilla_s4[i] = pc - range_val * 1.1 / 2
    
    # EMA34 for 1d trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current 6h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all 1d data to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to ensure previous day data exists
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R4 with volume spike in uptrend
            if (high[i] > camarilla_r4_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with volume spike in downtrend
            elif (low[i] < camarilla_s4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R4 or trend reversal
            if (close[i] < camarilla_r4_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S4 or trend reversal
            if (close[i] > camarilla_s4_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:13
