# Strategy: 12h_Camarilla_R3_S3_Breakout_1dEMA_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.384 | +36.3% | -6.2% | 67 | PASS |
| ETHUSDT | 0.018 | +20.5% | -9.3% | 61 | PASS |
| SOLUSDT | 0.314 | +42.6% | -21.0% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.021 | -2.8% | -8.4% | 29 | FAIL |
| ETHUSDT | 0.172 | +8.0% | -5.8% | 23 | PASS |
| SOLUSDT | -1.092 | -9.3% | -18.5% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dEMA_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    # Note: Using previous day's OHLC to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and levels from previous day's OHLC
    # Shift by 1 to use previous day's data
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN since no previous day exists
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    prev_daily_range = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r3 = pivot + 1.1 * prev_daily_range / 4
    s3 = pivot - 1.1 * prev_daily_range / 4
    
    # Align Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(ema34_12h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend and volume spike
            if close[i] > r3_12h[i] and close[i] > ema34_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend and volume spike
            elif close[i] < s3_12h[i] and close[i] < ema34_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S3 OR trend turns down
            if close[i] < s3_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R3 OR trend turns up
            if close[i] > r3_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 08:07
