#!/usr/bin/env python3
# 6h_ema_12h_swing_v1
# Hypothesis: 6h strategy using EMA(21) trend filter from 12h and swing failure points for entries.
# Enters long when 12h EMA(21) is rising (bullish trend), price makes a higher low above 6h EMA(50), and closes above prior swing high.
# Enters short when 12h EMA(21) is falling (bearish trend), price makes a lower high below 6h EMA(50), and closes below prior swing low.
# Uses swing points from 6h timeframe for entry timing and 12h EMA for trend direction.
# Works in bull/bear via trend filter and mean reversion to 6h EMA(50) during pullbacks.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_12h_swing_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h HTF data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate EMA(21) on 12h closes
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align 12h EMA(21) to 6h timeframe (completed 12h candle only)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 6h EMA(50) for dynamic support/resistance
    ema_50_6h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate swing points on 6h (pivot highs/lows)
    # Swing high: high[i] > high[i-1] and high[i] > high[i+1]
    # Swing low: low[i] < low[i-1] and low[i] < low[i+1]
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(1, n-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low[i] = True
    
    # Track most recent swing high and low
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high_val = high[i]
        if swing_low[i]:
            last_low_val = low[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(last_swing_high[i]) or np.isnan(last_swing_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h EMA trend: rising if current > previous, falling if current < previous
        if i > 0:
            ema_trend_rising = ema_21_12h_aligned[i] > ema_21_12h_aligned[i-1]
            ema_trend_falling = ema_21_12h_aligned[i] < ema_21_12h_aligned[i-1]
        else:
            ema_trend_rising = False
            ema_trend_falling = False
        
        if position == 1:  # Long position
            # Exit: price falls below 6h EMA(50) or makes lower low below prior swing low
            if close[i] < ema_50_6h[i] or (i > 0 and low[i] < last_swing_low[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 6h EMA(50) or makes higher high above prior swing high
            if close[i] > ema_50_6h[i] or (i > 0 and high[i] > last_swing_high[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: 12h EMA trending up, price above 6h EMA(50), and breaks above prior swing high
            if (ema_trend_rising and close[i] > ema_50_6h[i] and 
                i > 0 and close[i] > last_swing_high[i-1] and close[i-1] <= last_swing_high[i-1]):
                position = 1
                signals[i] = 0.25
            # Enter short: 12h EMA trending down, price below 6h EMA(50), and breaks below prior swing low
            elif (ema_trend_falling and close[i] < ema_50_6h[i] and 
                  i > 0 and close[i] < last_swing_low[i-1] and close[i-1] >= last_swing_low[i-1]):
                position = -1
                signals[i] = -0.25
    
    return signals