# Strategy: 4h_WilliamsAlligator_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.326 | +5.9% | -13.4% | 411 | FAIL |
| ETHUSDT | 0.072 | +22.8% | -10.3% | 380 | PASS |
| SOLUSDT | 0.213 | +33.4% | -26.0% | 371 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.176 | +8.0% | -8.0% | 128 | PASS |
| SOLUSDT | 0.664 | +16.7% | -7.8% | 122 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 12h EMA50 + Volume Spike
# Williams Alligator identifies trend direction via three SMAs (Jaw, Teeth, Lips).
# Lips (5 SMA) above Teeth (8 SMA) above Jaw (13 SMA) = bullish trend.
# Teeth above Jaw and Lips below Teeth = bearish trend.
# 12h EMA50 confirms multi-timeframe trend alignment.
# Volume spike (>1.5x 20-bar median) filters for institutional participation.
# Designed to catch strong trends while avoiding chop.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h: Jaw(13), Teeth(8), Lips(5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # 12h EMA50 for multi-timeframe trend confirmation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for all indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Bullish Alligator: Lips > Teeth > Jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish Alligator: Jaw > Teeth > Lips
        bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Long: Bullish Alligator + price above 12h EMA50 + volume spike
        if (bullish and 
            close[i] > ema_50_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bearish Alligator + price below 12h EMA50 + volume spike
        elif (bearish and 
              close[i] < ema_50_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Alligator reverses or price crosses 12h EMA50 in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish or close[i] < ema_50_12h_aligned[i])) or
               (signals[i-1] == -0.25 and (not bearish or close[i] > ema_50_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 07:32
