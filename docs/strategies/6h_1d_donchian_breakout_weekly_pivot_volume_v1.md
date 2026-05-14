# Strategy: 6h_1d_donchian_breakout_weekly_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.208 | +30.1% | -12.1% | 123 | PASS |
| ETHUSDT | 0.588 | +58.4% | -12.7% | 111 | PASS |
| SOLUSDT | 0.827 | +121.9% | -20.2% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.692 | -10.9% | -13.3% | 48 | FAIL |
| ETHUSDT | 0.272 | +9.7% | -9.5% | 41 | PASS |
| SOLUSDT | -0.475 | -3.1% | -14.7% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot alignment
    # Weekly pivot from previous 5 trading days provides institutional bias
    # Volume > 1.5x 20-period average confirms breakout validity
    # Target: 12-25 trades/year (50-100 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous week's pivot (using previous 5 trading days ~ 1 week)
    # Weekly high/low/close from 5d ago to 1d ago
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    for i in range(5, len(high_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])   # Previous 5 days high
        weekly_low[i] = np.min(low_1d[i-5:i])     # Previous 5 days low
        weekly_close[i] = close_1d[i-1]           # Previous day close
    
    # Weekly pivot points (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Get 6h Donchian(20) for breakout
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (1.5 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    donchian_high_aligned = donchian_high  # Already LTF
    donchian_low_aligned = donchian_low    # Already LTF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Weekly pivot bias
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Strong bullish/bearish conditions (beyond R1/S1)
        strong_bullish = close[i] > r1_aligned[i]
        strong_bearish = close[i] < s1_aligned[i]
        
        # Entry logic: Breakout + pivot alignment + volume confirmation
        long_entry = long_breakout and bullish_bias and strong_bullish and volume_spike_6h[i]
        short_entry = short_breakout and bearish_bias and strong_bearish and volume_spike_6h[i]
        
        # Exit logic: price returns to weekly pivot or opposite breakout
        long_exit = close[i] <= pivot_aligned[i] or short_breakout
        short_exit = close[i] >= pivot_aligned[i] or long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 00:20
