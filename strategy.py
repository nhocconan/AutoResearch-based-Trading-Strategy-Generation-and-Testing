#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
# Weekly pivot levels (PP, R1, R2, S1, S2) derived from previous week's high/low/close provide key institutional levels.
# Breakout above weekly R1 or below weekly S1 with volume spike (>1.5x 20-period average) and price positioned relative to weekly pivot
# captures institutional breakout moves while avoiding false signals. Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Weekly pivot filter ensures alignment with higher timeframe structure, working in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily data
    # Group daily data into weeks (starting Monday)
    # For each week: weekly_high = max(high), weekly_low = min(low), weekly_close = last close of week
    # We'll calculate weekly pivot using the most recent completed week
    
    # Create arrays for weekly pivot components
    weekly_high = np.full_like(close_1d, np.nan)
    weekly_low = np.full_like(close_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    # Simple approach: use rolling window of 5 days to approximate weekly
    # More accurate would require proper week grouping, but 5-day rolling captures weekly structure
    window = 5
    for i in range(window-1, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-window+1:i+1])
        weekly_low[i] = np.min(low_1d[i-window+1:i+1])
        weekly_close[i] = close_1d[i]  # last day of the week
    
    # Calculate weekly pivot levels
    # Pivot = (Weekly_High + Weekly_Low + Weekly_Close) / 3
    # R1 = (2 * Pivot) - Weekly_Low
    # S1 = (2 * Pivot) - Weekly_High
    # R2 = Pivot + (Weekly_High - Weekly_Low)
    # S2 = Pivot - (Weekly_High - Weekly_Low)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Calculate 20-period Donchian channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly pivot levels to 6h timeframe (waits for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot_val = weekly_pivot_aligned[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        r2_val = weekly_r2_aligned[i]
        s2_val = weekly_s2_aligned[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND above weekly R1 + volume spike
            if price > dch_high and price > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND below weekly S1 + volume spike
            elif price < dch_low and price < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or below weekly pivot
                if price < dch_low or price < pivot_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or above weekly pivot
                if price > dch_high or price > pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0