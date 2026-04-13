#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and choppiness regime filter.
# Long: price breaks above H3 + volume > 1.3x avg volume + chop > 61.8 (range)
# Short: price breaks below L3 + volume > 1.3x avg volume + chop > 61.8 (range)
# Camarilla levels calculated from 1d: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
# Chop filter avoids trending markets where breakouts fail; works in ranging markets.
# Position size: 0.25 to limit drawdown. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels H3 and L3 (using prior day's data)
    H3 = np.full(len(close_1d), np.nan)
    L3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        rng = high_1d[i-1] - low_1d[i-1]
        H3[i] = close_1d[i-1] + 1.1 * rng / 6.0
        L3[i] = close_1d[i-1] - 1.1 * rng / 6.0
    
    # Align 1d Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Average volume (20-period = 20*4h = ~3.3 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Choppiness Index (14-period) for regime filter
    chop = np.full(n, np.nan)
    for i in range(14, n):
        atr_sum = 0.0
        for j in range(i-13, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        # Chop filter: range-bound market (chop > 61.8)
        range_filter = chop_val > 61.8
        
        if position == 0:
            # Long: break above H3 + volume confirmation + range market
            if (price > H3_aligned[i] and 
                volume_confirm and
                range_filter):
                position = 1
                signals[i] = position_size
            # Short: break below L3 + volume confirmation + range market
            elif (price < L3_aligned[i] and 
                  volume_confirm and
                  range_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or chop < 38.2 (trending)
            if (price < L3_aligned[i] or
                chop_val < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or chop < 38.2 (trending)
            if (price > H3_aligned[i] or
                chop_val < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Camarilla_Chop_Volume"
timeframe = "4h"
leverage = 1.0