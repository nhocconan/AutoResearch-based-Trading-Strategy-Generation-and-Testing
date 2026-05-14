# Strategy: 4h_WilliamsAlligator_Volume_12hEMA

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.021 | +21.0% | -9.5% | 217 | PASS |
| ETHUSDT | 0.245 | +33.2% | -8.3% | 207 | PASS |
| SOLUSDT | 0.880 | +118.0% | -19.0% | 180 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.364 | -6.1% | -8.2% | 84 | FAIL |
| ETHUSDT | 0.211 | +8.6% | -8.6% | 69 | PASS |
| SOLUSDT | -0.300 | +0.7% | -10.8% | 66 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Volume Spike + 12h EMA Filter
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trends. 
# Long when Lips > Teeth > Jaw and price above 12h EMA50; short when Lips < Teeth < Jaw and price below 12h EMA50.
# Volume confirmation requires > 2x 20-bar median volume. 
# Designed to work in both bull (trend following) and bear (mean reversion via Alligator convergence).
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw (Blue)
    teeth = smma(close, 8)  # Teeth (Red)
    lips = smma(close, 5)   # Lips (Green)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Lips > Teeth > Jaw (bullish alignment), volume spike, price above 12h EMA50
        if (lips[i] > teeth[i] > jaw[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_12h_aligned[i]):
            signals[i] = 0.20
        
        # Short: Lips < Teeth < Jaw (bearish alignment), volume spike, price below 12h EMA50
        elif (lips[i] < teeth[i] < jaw[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_12h_aligned[i]):
            signals[i] = -0.20
        
        # Exit: Alligator convergence (Lips crosses Teeth/Jaw) or price crosses 12h EMA
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and (lips[i] <= teeth[i] or close[i] <= ema_12h_aligned[i])) or
               (signals[i-1] == -0.20 and (lips[i] >= teeth[i] or close[i] >= ema_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsAlligator_Volume_12hEMA"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 07:01
