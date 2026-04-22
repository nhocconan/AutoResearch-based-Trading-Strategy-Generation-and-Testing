#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 20-period high + weekly pivot support + volume spike
# Short when price breaks below 20-period low + weekly pivot resistance + volume spike
# Exit when price crosses opposite Donchian band or trend weakens
# Designed for low trade frequency (~15-35/year) to minimize fee drain and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point for bias
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 20-period Donchian channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: break above upper band + price above weekly pivot + volume spike
            if price > upper and price > weekly_pivot_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + price below weekly pivot + volume spike
            elif price < lower and price < weekly_pivot_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses opposite Donchian band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below lower band
                if price < lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above upper band
                if price > upper:
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