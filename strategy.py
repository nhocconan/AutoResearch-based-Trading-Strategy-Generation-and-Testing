#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Price Channel (Donchian 20-week) ===
    df_1w = get_htf_data(prices, '1w')
    highest = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to daily
    highest_aligned = align_htf_to_ltf(prices, df_1w, highest)
    lowest_aligned = align_htf_to_ltf(prices, df_1w, lowest)
    
    # === Volume Confirmation (daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure 20 weeks of data
    warmup = 20 * 7  # 20 weeks in days
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_aligned[i]) or np.isnan(lowest_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Close position when price returns to channel ===
        if position == 1:  # Long position
            # Exit when price crosses back below midline
            midline = (highest_aligned[i] + lowest_aligned[i]) / 2
            if price < midline:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above midline
            midline = (highest_aligned[i] + lowest_aligned[i]) / 2
            if price > midline:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-week high with volume confirmation
            if price > highest_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 20-week low with volume confirmation
            elif price < lowest_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1w_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0