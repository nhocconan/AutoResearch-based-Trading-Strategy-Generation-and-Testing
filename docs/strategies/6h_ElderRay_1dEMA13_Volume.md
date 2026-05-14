# Strategy: 6h_ElderRay_1dEMA13_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.820 | -15.8% | -27.9% | 604 | DISCARD |
| ETHUSDT | 0.074 | +22.6% | -11.2% | 598 | KEEP |
| SOLUSDT | 0.918 | +145.6% | -28.4% | 502 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.121 | +7.2% | -15.2% | 206 | KEEP |
| SOLUSDT | -0.451 | -2.7% | -14.5% | 179 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to detect strength of buyers/sellers
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In trending markets, Bull Power stays positive in uptrends, Bear Power stays positive in downtrends
# Volume confirmation filters weak breakouts
# EMA13 from 1d provides higher timeframe trend bias to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_ElderRay_1dEMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA13 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate EMA13 on 6h for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power: High - EMA13
    bear_power = ema_13_6h - low   # Bear Power: EMA13 - Low
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_6h[i]) or np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buyers in control) + above 1d EMA13 + volume confirmation
            if (bull_power[i] > 0 and 
                close[i] > ema_13_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (sellers in control) + below 1d EMA13 + volume confirmation
            elif (bear_power[i] > 0 and 
                  close[i] < ema_13_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bear Power becomes positive (sellers take control) or breaks below 1d EMA13
            if (bear_power[i] > 0) or (close[i] < ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bull Power becomes positive (buyers take control) or breaks above 1d EMA13
            if (bull_power[i] > 0) or (close[i] > ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 16:47
