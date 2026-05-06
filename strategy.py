#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1w Donchian upper channel (20) AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range regime)
# Short when price breaks below 1w Donchian lower channel (20) AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range regime)
# Exit when price retests the 1w Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 40-80 total trades over 4 years (10-20/year) for 1d timeframe
# 1w Donchian provides strong structural breakout levels with continuation probability
# Volume confirmation validates breakout strength while limiting false signals
# Choppiness regime filter ensures we only trade in ranging markets (mean reversion logic)
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by fading extreme moves in ranges

name = "1d_Donchian20_Volume_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed 1w bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no prior close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_regime = chop > 61.8  # Range regime (mean revert)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper, volume spike, in range regime
            if (close[i] > upper_aligned[i] and 
                volume_confirm[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower, volume spike, in range regime
            elif (close[i] < lower_aligned[i] and 
                  volume_confirm[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1w Donchian midpoint
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1w Donchian midpoint
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals