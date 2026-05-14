# Strategy: 1h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.563 | +54.6% | -11.7% | 132 | PASS |
| ETHUSDT | 0.162 | +28.7% | -12.5% | 167 | PASS |
| SOLUSDT | 0.911 | +156.2% | -21.9% | 176 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.513 | +0.1% | -5.3% | 62 | FAIL |
| ETHUSDT | 0.632 | +16.8% | -8.1% | 51 | PASS |
| SOLUSDT | 0.473 | +14.9% | -9.5% | 48 | PASS |

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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/4
    # S3 = C - (H - L) * 1.1/4
    pivot_1d = (high_1d + low_1d + close_1d_arr) / 3.0
    r3_1d = close_1d_arr + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = close_1d_arr - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align to 1h timeframe (previous day's levels available at open)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in uptrend with volume confirmation and during session
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in downtrend with volume confirmation and during session
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:53
