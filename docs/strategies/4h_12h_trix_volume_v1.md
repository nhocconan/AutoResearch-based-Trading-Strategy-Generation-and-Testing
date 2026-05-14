# Strategy: 4h_12h_trix_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.064 | +15.0% | -14.0% | 157 | FAIL |
| ETHUSDT | 0.150 | +27.8% | -12.1% | 145 | PASS |
| SOLUSDT | 1.153 | +229.7% | -18.0% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.168 | +8.1% | -10.3% | 54 | PASS |
| SOLUSDT | 0.387 | +12.8% | -9.8% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_12h_trix_volume_v1
# Strategy: 4h TRIX (Triple Exponential Average) momentum with volume confirmation and 12h trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX filters out insignificant price movements and shows the underlying trend momentum. 
# In bull markets, TRIX > 0 with rising momentum and volume confirmation. In bear markets, TRIX < 0 with falling momentum and volume confirmation.
# Uses 12h EMA50 for trend filter to avoid counter-trend trades. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_trix_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # TRIX calculation: Triple EMA of price, then 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(trix.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: TRIX momentum + volume + trend alignment
        if (trix.iloc[i] > 0 and trix.iloc[i] > trix.iloc[i-1] and  # TRIX positive and rising
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (trix.iloc[i] < 0 and trix.iloc[i] < trix.iloc[i-1] and  # TRIX negative and falling
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX momentum divergence or trend change
        elif position == 1 and (trix.iloc[i] <= 0 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix.iloc[i] >= 0 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 15:40
