# Strategy: 4h_Camarilla_R1S1_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.045 | +19.9% | -11.3% | 334 | FAIL |
| ETHUSDT | 0.296 | +31.6% | -10.0% | 303 | PASS |
| SOLUSDT | 0.396 | +42.5% | -17.4% | 249 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.475 | +10.8% | -4.2% | 115 | PASS |
| SOLUSDT | -0.179 | +4.2% | -6.5% | 91 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Volume
Breakout at Camarilla R1/S1 levels with 12h EMA50 trend filter and volume confirmation.
Long when price closes above R1 with bullish 12h trend (price > EMA50) and volume spike.
Short when price closes below S1 with bearish 12h trend (price < EMA50) and volume spike.
Exit when price returns to daily pivot point.
Designed for low trade frequency (20-40/year) to minimize fee drag in both bull and bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Load 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot levels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's OHLC for today's pivot levels
    # Shift by 1 to use previous day's data
    high_prev = np.roll(high_d, 1)
    low_prev = np.roll(low_d, 1)
    close_prev = np.roll(close_d, 1)
    # First day has no previous, set to NaN
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels (S1 and R1)
    s1 = close_prev - (range_val * 1.1 / 12)
    r1 = close_prev + (range_val * 1.1 / 12)
    
    # Align all levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA lookback
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price closes above R1 with bullish 12h trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price closes below S1 with bearish 12h trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot point
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot
                if close[i] <= pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot
                if close[i] >= pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%
```

## Last Updated
2026-04-22 17:04
