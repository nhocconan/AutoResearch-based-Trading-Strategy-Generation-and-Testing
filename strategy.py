#!/usr/bin/env python3
# 4h_ThreeDrive_Reversal_With_Volume
# Hypothesis: Three-drive patterns (higher highs in uptrend or lower lows in downtrend) 
# indicate exhaustion and impending reversal. We enter on the third drive's pullback 
# with volume confirmation and 1d trend filter. Works in both bull and bear markets 
# by trading reversals at trend extremes. Target: 15-30 trades/year.

name = "4h_ThreeDrive_Reversal_With_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and swing detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Identify swing highs and lows on 1d data
    # Swing high: high > previous 2 highs and next 2 highs
    # Swing low: low < previous 2 lows and next 2 lows
    swing_high = np.zeros(len(df_1d), dtype=bool)
    swing_low = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(2, len(df_1d) - 2):
        if (df_1d['high'].iloc[i] > df_1d['high'].iloc[i-1] and 
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i-2] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+1] and
            df_1d['high'].iloc[i] > df_1d['high'].iloc[i+2]):
            swing_high[i] = True
            
        if (df_1d['low'].iloc[i] < df_1d['low'].iloc[i-1] and 
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i-2] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+1] and
            df_1d['low'].iloc[i] < df_1d['low'].iloc[i+2]):
            swing_low[i] = True
    
    # Align swing signals to 4h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high.astype(float))
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low.astype(float))
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_highs = 0
    consecutive_lows = 0
    
    # Warmup: need 1d EMA50 (50), swing detection (need 2 days buffer), volume MA (20)
    start_idx = max(50, 20) + 2
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Update swing counters
        if swing_high_aligned[i] > 0.5:
            consecutive_highs += 1
            consecutive_lows = 0
        elif swing_low_aligned[i] > 0.5:
            consecutive_lows += 1
            consecutive_highs = 0
        else:
            # Decay counters slowly
            consecutive_highs = max(0, consecutive_highs - 0.1)
            consecutive_lows = max(0, consecutive_lows - 0.1)
        
        if position == 0:
            # Long entry: downtrend exhaustion (3+ swing lows) + pullback + volume
            if downtrend and consecutive_lows >= 3 and close[i] > low[i-1] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: uptrend exhaustion (3+ swing highs) + pullback + volume
            elif uptrend and consecutive_highs >= 3 and close[i] < high[i-1] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend resumes or reversal signal
            if uptrend or consecutive_highs >= 2:
                signals[i] = 0.0
                position = 0
                consecutive_highs = 0
                consecutive_lows = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend resumes or reversal signal
            if downtrend or consecutive_lows >= 2:
                signals[i] = 0.0
                position = 0
                consecutive_highs = 0
                consecutive_lows = 0
            else:
                signals[i] = -0.25
    
    return signals