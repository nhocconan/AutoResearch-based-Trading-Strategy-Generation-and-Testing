#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1d Donchian upper channel (20) AND volume > 1.5 * avg_volume(20) AND chop(14) > 61.8 (range regime)
# Short when price breaks below 1d Donchian lower channel (20) AND volume > 1.5 * avg_volume(20) AND chop(14) > 61.8 (range regime)
# Exit when price retests the 1d Donchian midpoint (median of upper/lower)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian breakouts provide strong structural levels with continuation probability
# Volume confirmation validates breakout strength
# Choppiness filter ensures we only trade in ranging markets where mean reversion at channel edges works
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by fading false breakouts in ranges

name = "4h_Donchian20_Volume_Chop"
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
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate Choppiness Index (14) on 4h data for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = sum of TR1
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * np.log10(highest_high - lowest_low + 1e-10)
    chop = 100 * np.log10(sum_atr1 + 1e-10) / chop_denominator
    chop_filter = chop > 61.8  # Range regime
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper, volume spike, in range regime, in session
            if (close[i] > upper_aligned[i] and 
                volume_confirm[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower, volume spike, in range regime, in session
            elif (close[i] < lower_aligned[i] and 
                  volume_confirm[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1d Donchian midpoint
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1d Donchian midpoint
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals