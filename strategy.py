#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v2
# Hypothesis: Uses weekly Donchian channel breakout for trend direction, with daily volume confirmation for entry timing. Enters long when price breaks above 20-week high with above-average volume, short when price breaks below 20-week low with above-average volume. Exits when price returns to the 20-week midpoint. Designed for 15-25 trades/year on daily timeframe to avoid fee drag. Works in bull/bear via trend-following with volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channel
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate 20-period Donchian channel on weekly data
    lookback = 20
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    
    for i in range(lookback, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i-lookback:i])
        donchian_low[i] = np.min(low_weekly[i-lookback:i])
    
    # Calculate average volume (50-period) for confirmation
    vol_avg = np.full_like(volume, np.nan)
    vol_lookback = 50
    for i in range(vol_lookback, len(volume)):
        vol_avg[i] = np.mean(volume[i-vol_lookback:i])
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Calculate midpoint for exit
    donchian_mid = (donchian_high_aligned + donchian_low_aligned) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        volume_confirmed = volume[i] > vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals