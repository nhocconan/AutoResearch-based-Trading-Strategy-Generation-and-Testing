#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation.
Long when price breaks above weekly Donchian high with expanding volume.
Short when price breaks below weekly Donchian low with expanding volume.
Exit when price crosses back to weekly Donchian midpoint.
Uses weekly Donchian channels for trend direction and daily breakouts for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels
    weekly_donch_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_donch_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    weekly_donch_mid = (weekly_donch_high + weekly_donch_low) / 2
    
    # Align to daily timeframe
    weekly_donch_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donch_high)
    weekly_donch_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donch_low)
    weekly_donch_mid_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donch_mid)
    
    # === Volume confirmation (daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(weekly_donch_high_aligned[i]) or np.isnan(weekly_donch_low_aligned[i]) or
            np.isnan(weekly_donch_mid_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below weekly midpoint
            if close[i] < weekly_donch_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above weekly midpoint
            if close[i] > weekly_donch_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Daily breakout of weekly Donchian channels with volume confirmation
            if close[i] > weekly_donch_high_aligned[i]:
                # Breakout above weekly Donchian high -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < weekly_donch_low_aligned[i]:
                # Breakdown below weekly Donchian low -> short
                position = -1
                signals[i] = -0.25
    
    return signals