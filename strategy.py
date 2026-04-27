#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Choppiness Index > 61.8 = ranging (mean revert at Donchian bands)
# Choppiness Index < 38.2 = trending (breakout in direction of price)
# Volume spike confirms breakout strength. Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        # Long conditions: Trending market (CHOP < 38.2) + price breaks above Donchian high + volume
        if (chop_val < 38.2 and 
            close[i] > highest_high[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Trending market (CHOP < 38.2) + price breaks below Donchian low + volume
        elif (chop_val < 38.2 and 
              close[i] < lowest_low[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Mean reversion in ranging market: CHOP > 61.8
        elif chop_val > 61.8:
            # Near lower band -> long
            if close[i] <= lowest_low[i] * 1.001:  # Within 0.1% of low
                signals[i] = 0.25
                position = 1
            # Near upper band -> short
            elif close[i] >= highest_high[i] * 0.999:  # Within 0.1% of high
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Transition zone or no clear signal
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Chop_Donchian_Breakout_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0