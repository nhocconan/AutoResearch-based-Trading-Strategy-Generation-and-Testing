# Strategy: 4h_12h_donchian_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +33.1% | -18.9% | 117 | PASS |
| ETHUSDT | 0.307 | +42.3% | -14.1% | 109 | PASS |
| SOLUSDT | 1.158 | +277.4% | -26.3% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.003 | -6.6% | -12.8% | 46 | FAIL |
| ETHUSDT | 0.852 | +24.4% | -7.8% | 35 | PASS |
| SOLUSDT | 0.438 | +14.9% | -12.9% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_12h_donchian_volume_v2
# Strategy: 4h Donchian breakout with volume confirmation and 12h EMA trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian(20) breakouts with volume confirmation and 12h EMA trend filter work in both bull and bear markets by capturing strong directional moves while avoiding whipsaws.
# Target: 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_20.iloc[i]) or 
            np.isnan(low_20.iloc[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        if vol_confirmed and close[i] > high_20.iloc[i-1] and close[i] > ema_50_12h_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif vol_confirmed and close[i] < low_20.iloc[i-1] and close[i] < ema_50_12h_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal or opposite breakout
        elif position == 1 and (close[i] < ema_50_12h_aligned[i] or close[i] < low_20.iloc[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_50_12h_aligned[i] or close[i] > high_20.iloc[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 17:09
