#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and chop regime filter
# Long when price breaks above 1d Donchian(20) upper band AND 12h volume > 1.2 * avg_volume(20) AND chop > 61.8 (range)
# Short when price breaks below 1d Donchian(20) lower band AND 12h volume > 1.2 * avg_volume(20) AND chop > 61.8 (range)
# Exit when price returns to 1d Donchian(20) midpoint
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 80-120 total trades over 4 years (20-30/year) for 12h timeframe
# 1d Donchian provides higher-timeframe structure, filtering out lower TF noise
# Volume confirmation filters out low-conviction breakouts
# Chop regime filter ensures we only trade breakouts from ranging markets (avoids trending chop)
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "12h_1dDonchian20_Breakout_Volume_Chop_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channel (20-period)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_1d = (upper_1d + lower_1d) / 2.0
    
    # Align 1d Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    # Calculate volume confirmation: volume > 1.2 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * avg_volume_20)
    
    # Calculate Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low))) / log10(n)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = true range
    sum_tr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(sum_tr1 / (14 * np.log10(chop_denom))) / np.log10(14)
    chop_filter = chop > 61.8  # range regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band, volume spike, chop > 61.8 (range)
            if (close[i] > upper_1d_aligned[i] and close[i-1] <= upper_1d_aligned[i-1] and 
                volume_confirm[i] and chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band, volume spike, chop > 61.8 (range)
            elif (close[i] < lower_1d_aligned[i] and close[i-1] >= lower_1d_aligned[i-1] and 
                  volume_confirm[i] and chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Donchian midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Donchian midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals