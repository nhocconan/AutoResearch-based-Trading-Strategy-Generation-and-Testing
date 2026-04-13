#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Choppiness Index regime filter + volume confirmation.
# Long: Price closes above Donchian(20) upper band + volume > 1.5x avg volume (20-period) + Chop(14) > 61.8 (range).
# Short: Price closes below Donchian(20) lower band + volume > 1.5x avg volume + Chop(14) > 61.8.
# Exit: Price crosses midline (average of upper/lower band) or Chop < 38.2 (trend regime).
# Uses Chop to filter for ranging markets where mean reversion works, volume to confirm breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

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
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
    
    atr_14 = np.zeros(len(tr1))
    for i in range(14, len(tr1)):
        atr_14[i] = np.mean(tr1[i-14:i])
    
    # Calculate Choppiness Index: 100 * log10(sum(TR14)/(ATR14 * n)) / log10(n)
    chop = np.full(len(close_1d), 50.0)  # default neutral
    for i in range(14, len(close_1d)):
        sum_tr = np.sum(tr1[i-14:i+1])  # including current
        if atr_14[i] > 0 and len(close_1d[i-14:i+1]) > 0:
            chop[i] = 100 * np.log10(sum_tr / (atr_14[i] * 14)) / np.log10(14)
    
    # Donchian(20) channels on 4h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        chop_val = chop_aligned[i]
        midline = (upper + lower) / 2
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        # Chop regime: > 61.8 = ranging (good for mean reversion)
        ranging = chop_val > 61.8
        # Trend regime: < 38.2 = trending (exit signal)
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: price closes above upper band + volume + ranging
            if (price > upper and volume_confirm and ranging):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower band + volume + ranging
            elif (price < lower and volume_confirm and ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below midline OR trend regime
            if (price < midline) or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above midline OR trend regime
            if (price > midline) or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Chop_Range_Breakout_Volume"
timeframe = "4h"
leverage = 1.0