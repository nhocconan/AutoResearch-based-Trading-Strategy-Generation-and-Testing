#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation
Long when price breaks above weekly Donchian high with volume spike
Short when price breaks below weekly Donchian low with volume spike
Exit when price returns to weekly Donchian midline
Designed to capture strong trends while avoiding chop with volume filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channels (20-period) ===
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate Donchian channels
    highest_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_daily = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # === Volume Spike Detector (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or np.isnan(volume_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midline or breaks below low
            if close[i] <= donchian_mid_daily[i] or close[i] < donchian_low_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midline or breaks above high
            if close[i] >= donchian_mid_daily[i] or close[i] > donchian_high_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry: Donchian breakout with volume confirmation
            if close[i] > donchian_high_daily[i] and volume_ratio[i] > 1.5:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low_daily[i] and volume_ratio[i] > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals