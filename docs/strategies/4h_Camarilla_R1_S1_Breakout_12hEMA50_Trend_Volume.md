# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.069 | +23.1% | -14.1% | 319 | PASS |
| ETHUSDT | 0.161 | +28.2% | -15.8% | 296 | PASS |
| SOLUSDT | 0.698 | +95.9% | -24.5% | 269 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.126 | -4.9% | -10.5% | 121 | FAIL |
| ETHUSDT | 0.948 | +21.7% | -6.5% | 109 | PASS |
| SOLUSDT | 0.885 | +20.8% | -10.9% | 95 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla R1 and S1 levels from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (using previous 1d bar's values)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 12h EMA50 (uptrend) AND volume surge
            if close[i] > r1_4h[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 AND below 12h EMA50 (downtrend) AND volume surge
            elif close[i] < s1_4h[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 12h EMA50 (trend change)
            if close[i] < s1_4h[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 12h EMA50 (trend change)
            if close[i] > r1_4h[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals
```

## Last Updated
2026-05-11 14:19
