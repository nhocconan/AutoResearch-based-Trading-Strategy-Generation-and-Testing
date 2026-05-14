# Strategy: 6h_Donchian20_WeeklyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.322 | +0.9% | -21.8% | 80 | FAIL |
| ETHUSDT | 0.164 | +28.8% | -15.1% | 77 | PASS |
| SOLUSDT | 0.726 | +120.8% | -27.5% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.223 | +9.3% | -11.3% | 26 | PASS |
| SOLUSDT | -0.048 | +3.1% | -16.0% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly Donchian channels (20-day high/low)
    # 20 trading days = approximately 4 weeks, but we'll use 20-day period for weekly context
    high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Use previous period's values to avoid look-ahead
    high_20d_prev = np.roll(high_20d, 1)
    low_20d_prev = np.roll(low_20d, 1)
    high_20d_prev[0] = np.nan
    low_20d_prev[0] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_20d_prev[i]) or 
            np.isnan(low_20d_prev[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume and above 1d EMA trend
            if close[i] > high_20d_prev[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume and below 1d EMA trend
            elif close[i] < low_20d_prev[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below 20-day low (trend reversal)
            if close[i] < low_20d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above 20-day high
            if close[i] > high_20d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 09:35
