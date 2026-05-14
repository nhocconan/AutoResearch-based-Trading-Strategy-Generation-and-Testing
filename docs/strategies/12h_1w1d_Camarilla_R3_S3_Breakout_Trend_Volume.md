# Strategy: 12h_1w1d_Camarilla_R3_S3_Breakout_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.560 | -8.3% | -24.0% | 41 | FAIL |
| ETHUSDT | 0.110 | +24.8% | -18.1% | 25 | PASS |
| SOLUSDT | 0.646 | +95.5% | -28.1% | 27 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.700 | +22.0% | -13.7% | 12 | PASS |
| SOLUSDT | -0.181 | -1.5% | -19.9% | 14 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_1w1d_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Combine 12h price action with 1w/1d trend and volume confirmation. 
Uses 1d EMA100 for trend filter and 1w EMA50 for higher timeframe bias. 
Breaks above R3 or below S3 of daily Camarilla levels with volume surge (>2x 24-bar average) 
and aligned weekly/daily trend. Designed for fewer, higher-quality trades (target: 15-30/year) 
to reduce fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # 1w EMA50 for higher timeframe bias
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 12h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filters: 
    # Daily trend: price > EMA100 = bullish, < EMA100 = bearish
    d1_uptrend = close > ema_100_1d_aligned
    d1_downtrend = close < ema_100_1d_aligned
    
    # Weekly trend bias: price > weekly EMA50 = bullish bias, < = bearish bias
    w1_bullish_bias = close > ema_50_1w_aligned
    w1_bearish_bias = close < ema_50_1w_aligned
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above R3 + daily uptrend + weekly bullish bias + volume surge
        long_entry = (close[i] > R3_aligned[i] and 
                     d1_uptrend[i] and 
                     w1_bullish_bias[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S3 + daily downtrend + weekly bearish bias + volume surge
        short_entry = (close[i] < S3_aligned[i] and 
                      d1_downtrend[i] and 
                      w1_bearish_bias[i] and 
                      volume_surge[i])
        
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

name = "12h_1w1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-28 04:48
