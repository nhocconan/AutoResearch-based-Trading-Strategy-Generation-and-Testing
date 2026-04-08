# Strategy: 4h_dual_ema_crossover_12h_volume_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.640 | -0.5% | -16.4% | 102 | FAIL |
| ETHUSDT | 0.050 | +22.0% | -15.6% | 103 | PASS |
| SOLUSDT | 0.906 | +124.0% | -15.5% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.543 | +14.0% | -9.9% | 37 | PASS |
| SOLUSDT | -0.352 | +0.5% | -12.5% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_dual_ema_crossover_12h_volume_filter_v1
Hypothesis: Dual EMA crossover (9/21) on 4h with volume confirmation and 12h EMA50 trend filter.
Enters long when fast EMA crosses above slow EMA with volume > 20-period average and price above 12h EMA50.
Enters short when fast EMA crosses below slow EMA with volume > 20-period average and price below 12h EMA50.
Uses dual EMA for reduced whipsaw vs single EMA, volume filter ensures momentum confirmation,
and 12h trend filter prevents counter-trend trades. Designed for 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_dual_ema_crossover_12h_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Dual EMA: fast 9, slow 21
    ema_fast = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # 12h trend filter
        above_12h_ema50 = close[i] > ema50_12h_aligned[i]
        below_12h_ema50 = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross down or price below 12h EMA50
            if ema_cross_down or below_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up or price above 12h EMA50
            if ema_cross_up or above_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: EMA cross up with volume confirmation and above 12h EMA50
            if ema_cross_up and vol_confirmed and above_12h_ema50:
                position = 1
                signals[i] = 0.25
            # Short: EMA cross down with volume confirmation and below 12h EMA50
            elif ema_cross_down and vol_confirmed and below_12h_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 20:36
