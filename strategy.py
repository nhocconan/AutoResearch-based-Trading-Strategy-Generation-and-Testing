#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Filter.
Long when price breaks above 20-period Donchian high with weekly uptrend and volume confirmation.
Short when price breaks below 20-period Donchian low with weekly downtrend and volume confirmation.
Exit when price crosses back below 10-period Donchian high (long) or above 10-period Donchian low (short).
Designed for 6h timeframe with ~50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_direction_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === DONCHIAN CHANNELS (6H) ===
    # 20-period for entry
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 10-period for exit (faster exit)
    donch_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # === VOLUME CONFIRMATION (6H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(one_w_ema_aligned[i]) or np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or
            np.isnan(donch_high_10[i]) or np.isnan(donch_low_10[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        uptrend = close[i] > one_w_ema_aligned[i]
        downtrend = close[i] < one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-period Donchian high OR trend turns down
            if close[i] < donch_high_10[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-period Donchian low OR trend turns up
            if close[i] > donch_low_10[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if close[i] > donch_high_20[i] and uptrend:
                # Breakout above 20-period high in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low_20[i] and downtrend:
                # Breakdown below 20-period low in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals