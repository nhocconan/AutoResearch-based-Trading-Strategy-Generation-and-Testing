# Strategy: 4H_Camarilla_R4_S4_Breakout_4hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.142 | +26.2% | -13.5% | 238 | PASS |
| ETHUSDT | 0.289 | +34.0% | -10.3% | 219 | PASS |
| SOLUSDT | 1.040 | +114.1% | -15.2% | 166 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.027 | -1.5% | -7.0% | 88 | FAIL |
| ETHUSDT | 0.240 | +8.6% | -6.2% | 80 | PASS |
| SOLUSDT | -0.345 | +1.7% | -10.7% | 65 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot points (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1/S1 and R4/S4)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + range_ * 1.1 / 12  # Resistance level 1
    s1 = close_1d - range_ * 1.1 / 12  # Support level 1
    r4 = close_1d + range_ * 1.1 / 2   # Resistance level 4
    s4 = close_1d - range_ * 1.1 / 2   # Support level 4
    
    # Align all levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume AND above EMA50 (uptrend)
            if (close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume AND below EMA50 (downtrend)
            elif (close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite R1/S1 level (tighter stop)
            if position == 1:
                if not np.isnan(s1_aligned[i]) and close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not np.isnan(r1_aligned[i]) and close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Camarilla_R4_S4_Breakout_4hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 13:11
