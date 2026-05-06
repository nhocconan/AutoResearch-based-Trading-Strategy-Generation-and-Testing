#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and chop regime filter
# Long when price breaks above 1d Donchian upper channel (20) AND chop > 61.8 (range) AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 1d Donchian lower channel (20) AND chop > 61.8 (range) AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price reaches the opposite Donchian band (long exits at lower, short exits at upper)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Donchian breakouts provide structural edge, chop filter ensures we only trade in ranging markets where mean reversion works
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading range extremes

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
    
    # Align 1d Donchian levels to 4h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1) over 14) / log10(highest high - lowest low over 14))
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # same length as high_1d
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = true range
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high - lowest_low)
    chop = 100 * (np.log10(sum_atr1) / chop_denom)
    chop = np.where(chop_denom <= 0, 50, chop)  # avoid division by zero or log of zero
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper, chop > 61.8 (range), volume spike, in session
            if (close[i] > upper_aligned[i] and 
                chop_aligned[i] > 61.8 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower, chop > 61.8 (range), volume spike, in session
            elif (close[i] < lower_aligned[i] and 
                  chop_aligned[i] > 61.8 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches 1d Donchian lower band (mean reversion target)
            if close[i] <= lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches 1d Donchian upper band (mean reversion target)
            if close[i] >= upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals