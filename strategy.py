#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Choppiness Index and Donchian Channel breakout.
# Long when price breaks above Donchian(20) upper band in trending market (CHOP < 38.2),
# Short when price breaks below Donchian(20) lower band in trending market (CHOP < 38.2).
# Exit when price returns to Donchian middle or market becomes choppy (CHOP > 61.8).
# Uses 1-day timeframe for Choppiness Index and Donchian Channel to avoid lower timeframe noise.
# Designed to work in both bull and bear markets by only trading in trending conditions.
# Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Choppiness Index and Donchian Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for CHOP(14) and Donchian(20)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    chop = 100 * np.log10(atr_14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate Donchian Channel (20)
    dc_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Align indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_middle_aligned = align_htf_to_ltf(prices, df_1d, dc_middle)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need CHOP and DC periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(dc_upper_aligned[i]) or
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(dc_middle_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        # Choppy market (CHOP > 61.8)
        choppy = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for Donchian breakouts in trending market
            # Long: price breaks above upper DC AND trending market
            if (close[i] > dc_upper_aligned[i] and 
                trending):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower DC AND trending market
            elif (close[i] < dc_lower_aligned[i] and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle DC or market becomes choppy
            if (close[i] <= dc_middle_aligned[i] or 
                choppy):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle DC or market becomes choppy
            if (close[i] >= dc_middle_aligned[i] or 
                choppy):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_Donchian_Breakout_v1"
timeframe = "4h"
leverage = 1.0