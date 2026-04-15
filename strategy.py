#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian breakout and volume confirmation
# Uses Choppiness Index (14) to detect ranging vs trending markets: 
# - CHOP > 61.8 = ranging (mean revert at Donchian channels)
# - CHOP < 38.2 = trending (breakout in direction of trend)
# Combines with Donchian(20) breakouts and volume confirmation for high-probability entries.
# Works in bull markets (breakouts up in trend) and bear markets (breakouts down in trend).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) on daily
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # Avoid division by zero
    
    # Align Choppiness Index to 4h timeframe (with 2-bar delay for confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # Calculate Donchian Channel (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x median of past 20 periods
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ma[i])):
            continue
        
        # Long conditions
        long_breakout = close[i] > highest_high_20[i]
        long_chop_trending = chop_aligned[i] < 38.2  # Trending market
        long_volume = volume[i] > 1.5 * volume_ma[i]
        
        # Short conditions
        short_breakout = close[i] < lowest_low_20[i]
        short_chop_trending = chop_aligned[i] < 38.2  # Trending market
        short_volume = volume[i] > 1.5 * volume_ma[i]
        
        # Mean reversion conditions (ranging market)
        long_mean_revert = (close[i] <= lowest_low_20[i] and 
                           chop_aligned[i] > 61.8 and  # Ranging market
                           volume[i] > 1.5 * volume_ma[i])
        short_mean_revert = (close[i] >= highest_high_20[i] and 
                            chop_aligned[i] > 61.8 and  # Ranging market
                            volume[i] > 1.5 * volume_ma[i])
        
        # Entry logic
        if position <= 0 and long_breakout and long_chop_trending and long_volume:
            position = 1
            signals[i] = base_size
        elif position >= 0 and short_breakout and short_chop_trending and short_volume:
            position = -1
            signals[i] = -base_size
        elif position <= 0 and long_mean_revert:
            position = 1
            signals[i] = base_size
        elif position >= 0 and short_mean_revert:
            position = -1
            signals[i] = -base_size
        
        # Exit logic
        elif position == 1 and (short_breakout or 
                               (chop_aligned[i] > 61.8 and close[i] >= highest_high_20[i]) or
                               (chop_aligned[i] < 38.2 and close[i] <= lowest_low_20[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or 
                                (chop_aligned[i] > 61.8 and close[i] <= lowest_low_20[i]) or
                                (chop_aligned[i] < 38.2 and close[i] >= highest_high_20[i])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_Breakout_MeanRev"
timeframe = "4h"
leverage = 1.0