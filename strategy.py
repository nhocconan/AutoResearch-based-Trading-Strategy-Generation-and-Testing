#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with volume confirmation and chop regime filter.
# Enter long when price breaks above 12h Donchian upper band with volume spike and chop < 61.8 (trending).
# Enter short when price breaks below 12h Donchian lower band with volume spike and chop < 61.8.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
# Donchian breakouts work in both bull and bear markets when combined with volume and regime filters.

name = "4h_Donchian20_12h_VolumeChop_v1"
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
    
    # Get 12h data for Donchian and chop calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    n_12h = len(high_12h)
    upper_20 = np.full(n_12h, np.nan)
    lower_20 = np.full(n_12h, np.nan)
    
    for i in range(n_12h):
        if i >= 19:  # Need 20 periods for Donchian
            upper_20[i] = np.max(high_12h[i-19:i+1])
            lower_20[i] = np.min(low_12h[i-19:i+1])
    
    # Forward fill Donchian levels
    upper_20 = pd.Series(upper_20).ffill().values
    lower_20 = pd.Series(lower_20).ffill().values
    
    # Align 12h Donchian levels to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Calculate 12h Chopiness Index (14-period) for regime filter
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr_12h = np.zeros(n_12h)
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, n_12h):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    # Sum of True Range over 14 periods
    atr_14 = np.zeros(n_12h)
    for i in range(n_12h):
        if i >= 13:  # Need 14 periods for ATR
            atr_14[i] = np.sum(tr_12h[i-13:i+1])
    
    # Chopiness Index = 100 * log10(sum(TR14) / (n * max(HH-LL))) / log10(n)
    chop = np.full(n_12h, 50.0)  # Default to neutral
    for i in range(n_12h):
        if i >= 33:  # Need 14 ATR + 14 HH/LL lookback
            sum_tr14 = atr_14[i]
            highest_high = np.max(high_12h[i-13:i+1])
            lowest_low = np.min(low_12h[i-13:i+1])
            max_range = highest_high - lowest_low
            if max_range > 0 and sum_tr14 > 0:
                chop[i] = 100 * np.log10(sum_tr14 / max_range) / np.log10(14)
    
    # Forward fill Chop
    chop = pd.Series(chop).ffill().values
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 indicates trending market (good for breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > upper_20_aligned[i] and volume_spike[i]
        short_breakout = close[i] < lower_20_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or chop > 61.8 (rangy market)
        long_exit = close[i] < lower_20_aligned[i] or chop_aligned[i] >= 61.8
        short_exit = close[i] > upper_20_aligned[i] or chop_aligned[i] >= 61.8
        
        # Handle entries and exits
        if long_breakout and trending_regime and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and trending_regime and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals