#!/usr/bin/env python3
"""
1D Donchian Breakout with Weekly Trend Filter and Volume Confirmation
Long when price breaks above 20-day Donchian high with volume > 20-day average AND weekly trend up
Short when price breaks below 20-day Donchian low with volume > 20-day average AND weekly trend down
Exit when price crosses back to 20-day EMA midpoint
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # === Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === EMA 20 for exit (midpoint) ===
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly trend filter (EMA 21) ===
    df_weekly = get_htf_data(prices, '1w')
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA midpoint
            if close[i] < ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA midpoint
            if close[i] > ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be above average
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND weekly trend filter
            if close[i] > donchian_high[i] and ema_weekly_aligned[i] > ema_weekly_aligned[i-1]:
                # Breakout above Donchian high with rising weekly EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and ema_weekly_aligned[i] < ema_weekly_aligned[i-1]:
                # Breakdown below Donchian low with falling weekly EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals