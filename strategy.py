#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Supertrend for direction and 1d volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) to determine trend direction and only takes
# long/short entries aligned with that trend. 1d volume filter requires current volume
# > 1.5x 20-period average volume to ensure institutional participation. Entry occurs
# on pullbacks to the Supertrend line. Designed for 15-35 trades/year with tight entries.
# Session filter (08-20 UTC) reduces noise. Position size fixed at 0.20.

name = "1h_4h_supertrend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(10)
    atr = np.full_like(close_4h, np.nan, dtype=float)
    for i in range(9, len(tr)):
        atr[i] = np.nanmean(tr[i-9:i+1])
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    
    supertrend = np.full_like(close_4h, np.nan, dtype=float)
    dir = np.full_like(close_4h, 1, dtype=float)  # 1 = uptrend, -1 = downtrend
    
    supertrend[0] = hl2[0]
    dir[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper[i-1]:
            dir[i] = 1
        elif close_4h[i] < lower[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if dir[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        if dir[i] == 1:
            supertrend[i] = lower[i]
        else:
            supertrend[i] = upper[i]
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    dir_aligned = align_htf_to_ltf(prices, df_4h, dir)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(dir_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine trend direction from Supertrend
        is_uptrend = dir_aligned[i] == 1
        is_downtrend = dir_aligned[i] == -1
        
        # Entry conditions: pullback to Supertrend in direction of trend
        enter_long = (low[i] <= supertrend_aligned[i] and vol_filter and is_uptrend)
        enter_short = (high[i] >= supertrend_aligned[i] and vol_filter and is_downtrend)
        
        # Exit conditions: close crosses Supertrend in opposite direction
        exit_long = (position == 1 and close[i] < supertrend_aligned[i])
        exit_short = (position == -1 and close[i] > supertrend_aligned[i])
        
        # Update position and signal
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals