# Strategy: 4H_Camarilla_S1S3_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.077 | +19.2% | -9.0% | 344 | FAIL |
| ETHUSDT | 0.227 | +28.8% | -9.3% | 298 | PASS |
| SOLUSDT | 0.488 | +48.4% | -15.8% | 254 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.113 | +7.0% | -5.0% | 108 | PASS |
| SOLUSDT | -0.154 | +4.5% | -7.9% | 93 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Point S1/S3 breakout with 12h trend filter and volume confirmation.
Long when price breaks above S3 with bullish 12h trend and volume spike.
Short when price breaks below S1 with bearish 12h trend and volume spike.
Exit when price returns to pivot point (P) or trend weakens.
Uses 12h EMA50 for trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (20-40/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate daily data for Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    pivot_d = (high_d + low_d + close_d) / 3.0
    range_d = high_d - low_d
    
    # Camarilla levels
    s1_d = close_d - (range_d * 1.1 / 12)
    s3_d = close_d - (range_d * 1.1 / 4)
    r1_d = close_d + (range_d * 1.1 / 12)
    r3_d = close_d + (range_d * 1.1 / 4)
    
    # Align all levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_d)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_d)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_d)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S3 with bullish 12h trend and volume spike
            if (close[i] > s3_aligned[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with bearish 12h trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to S1 OR trend turns bearish
                if close[i] <= s1_aligned[i] or close[i] < ema50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to R1 OR trend turns bullish
                if close[i] >= r1_aligned[i] or close[i] > ema50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_S1S3_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%
```

## Last Updated
2026-04-22 16:38
