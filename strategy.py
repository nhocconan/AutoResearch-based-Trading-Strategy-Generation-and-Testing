#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume confirmation + 1d chop regime filter
# Uses Donchian(20) on 4h for breakout signals, confirmed by volume spike (>1.5x 20-bar avg)
# Only trades in trending regimes as identified by 1d Choppiness Index (CHOP < 38.2)
# In choppy markets (CHOP >= 38.2), remains flat to avoid whipsaws
# Position size 0.25 to limit drawdown during 2022 bear market
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: trend filter prevents trading in choppy/range-bound markets

name = "4h_1d_donchian_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Sum of TR over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        tr_sum_14[i] = np.sum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high_14 = np.full(len(df_1d), np.nan)
    min_low_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        max_high_14[i] = np.max(high_1d[i-13:i+1])
        min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if max_high_14[i] > min_low_14[i]:
            chop_1d[i] = 100 * np.log10(tr_sum_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
    
    # Align 1d chop regime to 4h timeframe
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4h volume average (20-period)
    volume_ma = np.full(n, np.nan)
    for i in range(19, n):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(chop_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regimes (CHOP < 38.2)
        if chop_1d_4h[i] >= 38.2:
            # Choppy market - exit any position and stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trending regime - look for Donchian breakouts with volume confirmation
        if position == 1:  # Long position
            # Exit conditions: price crosses below midline OR volume drops significantly
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midline or volume[i] < 0.5 * volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses above midline OR volume drops significantly
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midline or volume[i] < 0.5 * volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with volume confirmation
            # Volume must be > 1.5x average to confirm breakout strength
            if volume[i] > 1.5 * volume_ma[i]:
                if close[i] > highest_high[i]:
                    # Bullish breakout
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i]:
                    # Bearish breakout
                    position = -1
                    signals[i] = -0.25
    
    return signals