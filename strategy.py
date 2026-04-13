#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Choppiness Index regime filter and Donchian breakout.
# Long: Price breaks above Donchian upper band (20-period) + 1d Choppiness Index > 61.8 (range) + volume > 1.3x average.
# Short: Price breaks below Donchian lower band (20-period) + 1d Choppiness Index > 61.8 (range) + volume > 1.3x average.
# Uses chop filter to avoid whipsaw in trends, Donchian for breakouts in ranging markets, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = np.abs(high_1d[i] - close_1d[i-1])
        lc = np.abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate Choppiness Index (14-period)
    chop = np.full(len(close_1d), np.nan)
    atr_sum = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        atr_sum[i] = np.sum(tr[i-13:i+1])  # Sum of TR for last 14 periods
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral if calculation invalid
    
    # Donchian channels (20-period) on 12h
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Choppiness Index to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # Chop filter: range market (Choppiness > 61.8)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above upper Donchian + chop filter + volume confirmation
            if (price > upper[i] and chop_filter and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian + chop filter + volume confirmation
            elif (price < lower[i] and chop_filter and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower Donchian
            if price < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Donchian
            if price > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Chop_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0