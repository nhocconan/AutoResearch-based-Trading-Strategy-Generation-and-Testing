# Strategy: 4h_daily_pivot_breakout_1d_trend_volume_v7

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.305 | +1.1% | -19.7% | 316 | FAIL |
| ETHUSDT | 0.046 | +19.5% | -18.2% | 308 | PASS |
| SOLUSDT | 0.782 | +137.2% | -22.2% | 290 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.164 | +8.0% | -12.3% | 101 | PASS |
| SOLUSDT | 0.046 | +5.4% | -14.6% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_pivot_breakout_1d_trend_volume_v7"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (previous day's values)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h trend: 34-period EMA (faster than 50 for better responsiveness)
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34[i]) or np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S1 or trend fails
            if close[i] < s1_4h[i] or close[i] < ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R1 or trend fails
            if close[i] > r1_4h[i] or close[i] > ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34[i]
            bearish = close[i] < ema_34[i]
            
            # Long: price > R1 + bullish trend + volume
            if (close[i] > r1_4h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S1 + bearish trend + volume
            elif (close[i] < s1_4h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 02:48
