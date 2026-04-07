#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend and Volume Confirmation.
Long when price breaks above 4-period high with 12h uptrend and volume confirmation.
Short when price breaks below 4-period low with 12h downtrend and volume confirmation.
Exit when price crosses back below entry level or trend reverses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # === 12h Trend Filter (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    twelve_h_close = df_12h['close'].values
    twelve_h_ema = pd.Series(twelve_h_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    twelve_h_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema)
    
    # === Donchian Channel (4-period high/low) ===
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    # === Volume Confirmation (4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_level = 0.0  # Track entry level for exit
    
    for i in range(20, n):
        if (np.isnan(twelve_h_ema_aligned[i]) or np.isnan(high_4[i]) or np.isnan(low_4[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > twelve_h_ema_aligned[i]
        downtrend = close[i] < twelve_h_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below entry level OR trend turns down
            if close[i] < entry_level or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above entry level OR trend turns up
            if close[i] > entry_level or uptrend:
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
            if close[i] > high_4[i] and uptrend:
                # Breakout above 4-period high in uptrend -> long
                position = 1
                entry_level = low_4[i]  # Exit if price falls back below 4-period low
                signals[i] = 0.25
            elif close[i] < low_4[i] and downtrend:
                # Breakdown below 4-period low in downtrend -> short
                position = -1
                entry_level = high_4[i]  # Exit if price rises back above 4-period high
                signals[i] = -0.25
    
    return signals