# Strategy: 4h_Camarilla_R3_S3_Breakout_12hEMA_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.109 | +25.0% | -9.7% | 208 | PASS |
| ETHUSDT | 0.161 | +28.2% | -12.6% | 199 | PASS |
| SOLUSDT | 0.676 | +88.7% | -30.7% | 167 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.979 | -3.8% | -6.8% | 74 | FAIL |
| ETHUSDT | 1.140 | +25.3% | -9.8% | 66 | PASS |
| SOLUSDT | -0.506 | -2.0% | -12.1% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Camarilla R3 and S3 levels (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and levels from daily OHLC
    daily_high = high_1d
    daily_low = low_1d
    daily_close = close_1d
    
    daily_range = daily_high - daily_low
    pivot = (daily_high + daily_low + daily_close) / 3
    r3 = pivot + 1.1 * daily_range / 4
    s3 = pivot - 1.1 * daily_range / 4
    
    # Camarilla levels for each daily bar
    camarilla_r3_1d = r3
    camarilla_s3_1d = s3
    
    # Align Camarilla levels and EMA to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend and volume spike
            if close[i] > r3_4h[i] and close[i] > ema50_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend and volume spike
            elif close[i] < s3_4h[i] and close[i] < ema50_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S3 OR trend turns down
            if close[i] < s3_4h[i] or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R3 OR trend turns up
            if close[i] > r3_4h[i] or close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 08:06
