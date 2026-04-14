#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index regime filter and Donchian breakout.
# Long when price breaks above Donchian upper band (20-period) and CHOP > 61.8 (range regime) for mean reversion.
# Short when price breaks below Donchian lower band and CHOP > 61.8 (range regime).
# Exit when price returns to Donchian middle (average of high/low) or CHOP < 38.2 (trending regime).
# Uses Choppiness Index to identify range-bound markets where mean reversion works, and Donchian channels for breakout signals.
# Designed to work in both bull and bear markets by focusing on range-bound conditions where price oscillates between levels.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Choppiness Index and Donchian channels
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
    
    # ATR (14-period smoothed TR)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh - ll)) / np.log10(14)
    
    # Donchian channels (20-period)
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
    start = max(34, 20)  # Need CHOP and Donchian periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(dc_upper_aligned[i]) or
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(dc_middle_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: CHOP > 61.8 indicates ranging market (good for mean reversion)
        range_regime = chop_aligned[i] > 61.8
        
        # Trending regime: CHOP < 38.2 indicates trending market (exit positions)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Look for Donchian breakouts in range regime
            # Long: price breaks above upper DC AND range regime
            if (close[i] > dc_upper_aligned[i] and range_regime):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower DC AND range regime
            elif (close[i] < dc_lower_aligned[i] and range_regime):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle DC or market starts trending
            if (close[i] <= dc_middle_aligned[i] or trending_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle DC or market starts trending
            if (close[i] >= dc_middle_aligned[i] or trending_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_Donchian_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0