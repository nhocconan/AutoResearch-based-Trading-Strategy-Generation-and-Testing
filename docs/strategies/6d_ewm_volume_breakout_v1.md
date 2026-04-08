# Strategy: 6d_ewm_volume_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.105 | +13.8% | -9.1% | 378 | FAIL |
| ETHUSDT | -0.250 | +1.8% | -18.2% | 381 | FAIL |
| SOLUSDT | 0.703 | +107.9% | -32.1% | 387 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.246 | +9.6% | -12.9% | 113 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_ewm_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EWM trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EWM (exponential weighted mean) with span=30
    close_1d = df_1d['close'].values
    ewm_1d = pd.Series(close_1d).ewm(span=30, adjust=False).mean().values
    
    # Align EWM to 6h timeframe
    ewm_1d_aligned = align_htf_to_ltf(prices, df_1d, ewm_1d)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price change momentum (6-period ROC on 6h)
    roc_6 = np.zeros(n)
    for i in range(6, n):
        roc_6[i] = (close[i] - close[i-6]) / close[i-6] if close[i-6] != 0 else 0
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ewm_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(roc_6[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to daily EWM
        price_above_ewm = close[i] > ewm_1d_aligned[i]
        price_below_ewm = close[i] < ewm_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Momentum confirmation
        mom_up = roc_6[i] > 0.005  # 0.5% momentum up
        mom_down = roc_6[i] < -0.005  # 0.5% momentum down
        
        # Exit conditions: opposite momentum
        exit_long = roc_6[i] < -0.003
        exit_short = roc_6[i] > 0.003
        
        if position == 1:  # Long position
            # Exit on negative momentum
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on positive momentum
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above daily EWM + volume + upward momentum
            if price_above_ewm and vol_confirm and mom_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price below daily EWM + volume + downward momentum
            elif price_below_ewm and vol_confirm and mom_down:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 06:32
