#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1d High-Low Range Breakout with Volume Confirmation and Choppiness Filter
# Uses the previous day's high-low range to establish a breakout channel
# Breaks above previous day's high or below previous day's low trigger entries
# Volume confirmation (>1.5x average) ensures significant participation
# Choppiness filter (CHOP > 61.8) avoids false breakouts in ranging markets
# Designed to capture momentum moves in both bull and bear markets
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for range and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d high, low, close for range breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for Choppiness indicator
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Calculate Choppiness Index (14-period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align 1d data to 4h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 61.8)
        trending = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above previous day's high with volume filter and trending market
            if price > high_1d_aligned[i] and vol > 1.5 * avg_vol[i] and trending:
                position = 1
                signals[i] = position_size
            # Short: price breaks below previous day's low with volume filter and trending market
            elif price < low_1d_aligned[i] and vol > 1.5 * avg_vol[i] and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below previous day's low or chop becomes too high
            if price < low_1d_aligned[i] or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above previous day's high or chop becomes too high
            if price > high_1d_aligned[i] or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_High_Low_Range_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0