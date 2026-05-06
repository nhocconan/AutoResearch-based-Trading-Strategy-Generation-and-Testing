#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and chop regime filter
# Long when price breaks above 1d Donchian upper channel AND chop > 61.8 (range) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Donchian lower channel AND chop > 61.8 (range) AND volume > 1.5 * avg_volume(20)
# Exit when price retraces to 1d Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Chop regime filter reduces whipsaw in strong trends, focusing on mean-reversion breaks in ranging markets
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by fading extreme moves in chop

name = "4h_Donchian20_Chop_Volume_MR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align 1d Donchian levels to 4h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
    
    # Calculate 14-period chopiness index on 1d timeframe for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_14 - low_14).rolling(window=14, min_periods=14).sum().values
    true_range = pd.Series(np.maximum(high_1d - low_1d, 
                                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                              np.abs(low_1d - np.roll(close_1d, 1))))).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_14 / true_range) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper AND chop > 61.8 (range) AND volume spike
            if (close[i] > upper_aligned[i] and 
                chop_aligned[i] > 61.8 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower AND chop > 61.8 (range) AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  chop_aligned[i] > 61.8 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retraces to 1d Donchian midpoint
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retraces to 1d Donchian midpoint
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals