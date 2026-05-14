# Strategy: 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Strict_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.140 | +27.0% | -13.5% | 136 | PASS |
| ETHUSDT | 0.574 | +75.0% | -19.6% | 109 | PASS |
| SOLUSDT | 0.826 | +169.1% | -34.4% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.801 | -5.6% | -12.5% | 50 | FAIL |
| ETHUSDT | 0.503 | +16.3% | -15.0% | 41 | PASS |
| SOLUSDT | 0.862 | +28.3% | -15.8% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Strict_v2
Hypothesis: Focus on high-probability breakouts with strict volume confirmation and trend alignment.
Uses 1d EMA50 for trend (more responsive than EMA200) and 2.0x volume threshold to balance signal frequency.
Targets 20-40 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels - R3 and S3 for breakouts
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA50 for trend filter (more responsive than EMA200)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d data to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 1d EMA50 trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and d1_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and d1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Strict_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 04:45
