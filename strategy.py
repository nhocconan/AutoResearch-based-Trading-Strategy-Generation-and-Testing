#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# Choppiness Index identifies trending vs ranging markets to avoid false breakouts in sideways periods
# Donchian breakout captures momentum in trending regimes, filtered by volume to avoid noise
# Designed for 4h timeframe targeting 20-30 trades/year with robust performance in bull/bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index (calculated once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop = np.where(max_range_14 != 0, 
                    100 * np.log10(sum_tr14 / max_range_14) / np.log10(14), 
                    50)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only enter in trending market (CHOP < 38.2) with breakout and volume
            if chop_aligned[i] < 38.2:  # Trending regime
                # Long: Donchian breakout above upper band + volume confirmation
                if (close[i] > highest_20[i-1] and  # breakout above previous period's high
                    volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                    signals[i] = 0.25
                    position = 1
                # Short: Donchian breakout below lower band + volume confirmation
                elif (close[i] < lowest_20[i-1] and   # breakout below previous period's low
                      volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: price returns to opposite Donchian band or chop increases (rangy market)
            if position == 1:
                # Exit long: price returns to lower Donchian band or market becomes choppy
                if (close[i] < lowest_20[i] or 
                    chop_aligned[i] > 61.8):  # Choppy regime
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to upper Donchian band or market becomes choppy
                if (close[i] > highest_20[i] or 
                    chop_aligned[i] > 61.8):  # Choppy regime
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Donchian20_Volume_Trend"
timeframe = "4h"
leverage = 1.0