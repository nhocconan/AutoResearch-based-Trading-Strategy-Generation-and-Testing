# 1d Weekly Donchian Breakout with Volume Confirmation
# Hypothesis: Weekly Donchian channel breakouts capture strong trend moves. Volume confirmation filters false breakouts.
# Works in bull markets (upward breakouts) and bear markets (downward breakdowns) by following price action.
# Limited to 1d timeframe to reduce trade frequency and avoid fee drag.
# Target: 30-100 trades over 4 years (7-25/year) to stay within fee drag limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over past 20 weeks
    donchian_high = np.full_like(high_1w, np.nan)
    for i in range(len(high_1w)):
        if i < 19:
            donchian_high[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1w[i-19:i+1])
    
    # Lower band: lowest low over past 20 weeks
    donchian_low = np.full_like(low_1w, np.nan)
    for i in range(len(low_1w)):
        if i < 19:
            donchian_low[i] = np.nan
        else:
            donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align to daily timeframe (use previous week's levels to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike
            if close[i] > donchian_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike
            elif close[i] < donchian_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly Donchian low or volume dies
            if close[i] < donchian_low_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly Donchian high or volume dies
            if close[i] > donchian_high_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0