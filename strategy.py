#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following strategy using 4h Supertrend for direction and 1h volume breakout for entry.
# Long when 4h Supertrend is bullish AND 1h price breaks above 20-bar high AND volume > 1.5x 20-bar average.
# Short when 4h Supertrend is bearish AND 1h price breaks below 20-bar low AND volume > 1.5x 20-bar average.
# Exit when 4h Supertrend flips direction.
# Uses 4h for signal direction (reducing trade frequency) and 1h for precise entry timing.
# Volume breakout filters for institutional participation. Target: 60-150 total trades over 4 years.

name = "1h_Supertrend4h_VolumeBreakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 4h Supertrend calculation (ATR=10, multiplier=3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Supertrend direction (1 for uptrend, -1 for downtrend)
    supertrend_direction = direction
    
    # Align 4h Supertrend direction to 1h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction)
    
    # 1h indicators: 20-bar high/low and volume filter
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for 20-bar calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h uptrend, price breaks above 20-bar high, volume filter
            long_cond = (supertrend_dir_aligned[i] == 1) and (close[i] > high_roll_max[i]) and volume_filter[i]
            # Short conditions: 4h downtrend, price breaks below 20-bar low, volume filter
            short_cond = (supertrend_dir_aligned[i] == -1) and (close[i] < low_roll_min[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns bearish
            if supertrend_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend turns bullish
            if supertrend_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals