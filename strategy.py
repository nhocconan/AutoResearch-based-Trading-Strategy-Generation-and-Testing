#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and Choppiness Index regime filter.
# Long when price breaks above 12h Donchian upper (20-period) AND volume > 1.5x 20-period average AND Choppiness > 61.8 (ranging market).
# Short when price breaks below 12h Donchian lower (20-period) AND volume > 1.5x 20-period average AND Choppiness > 61.8.
# Exit when price returns to 12h Donchian midpoint.
# Designed for ranging markets where mean reversion at channel extremes works well.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian, volume average, and Choppiness
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    high_series = pd.Series(high_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    low_series = pd.Series(low_12h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 12h volume average (20-period)
    volume_series = pd.Series(volume_12h)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (14-period)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period)
    atr = np.full_like(tr, np.nan)
    atr[13] = np.nanmean(tr[1:14])  # First ATR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over last 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over last 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(ATR) / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    chop = np.full_like(atr_sum, 50.0)  # Default to neutral
    valid = (hl_range > 0) & ~np.isnan(atr_sum)
    chop[valid] = 100 * np.log10(atr_sum[valid] / hl_range[valid]) / np.log10(14)
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    volume_avg_aligned = align_htf_to_ltf(prices, df_12h, volume_avg)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need Donchian and Chop periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or
            np.isnan(volume_avg_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > (1.5 * volume_avg_aligned[i])
        
        # Regime filter: Choppiness > 61.8 indicates ranging market
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for mean reversion entries in ranging market
            # Long: price breaks above Donchian upper AND volume confirm AND ranging market
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirm and 
                ranging_market):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower AND volume confirm AND ranging market
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirm and 
                  ranging_market):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if close[i] >= donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if close[i] <= donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0