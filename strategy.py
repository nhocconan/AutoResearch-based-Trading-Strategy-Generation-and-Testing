#!/usr/bin/env python3
"""
Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
Long when price breaks above 20-day Donchian high with above-average volume AND weekly trend up
Short when price breaks below 20-day Donchian low with above-average volume AND weekly trend down
Exit when price crosses back to Donchian middle (10-day average of high/low)
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
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
    
    # === Donchian Channels (20-period) ===
    # Upper: highest high of last 20 days
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower: lowest low of last 20 days
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle: average of upper and lower
    donch_mid = (donch_high + donch_low) / 2
    
    # === Volume confirmation (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Weekly trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need above-average volume (at least 1.2x)
            if vol_ratio[i] < 1.2:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND weekly trend filter
            if close[i] > donch_high[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                # Breakout above upper channel with rising weekly EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                # Breakdown below lower channel with falling weekly EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals