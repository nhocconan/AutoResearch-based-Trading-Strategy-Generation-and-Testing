#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + weekly Donchian breakout with volume confirmation
# Uses weekly Donchian(20) breakout for trend direction, 1d Choppiness Index > 61.8 for range filter,
# and volume spike (1.5x 20-period EMA) for confirmation. Designed for low-frequency trades
# (<100 total over 4 years) to minimize fee drag and work in both bull/bear markets.
# Choppiness Index > 61.8 indicates ranging market where breakouts fail; < 38.2 indicates trending.
# We use > 61.8 to avoid false breakouts in ranging conditions.

name = "1d_Chop_Donchian20w_Volume"
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
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower bands
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Get daily data for Choppiness Index
    high_1d = high
    low_1d = low
    close_1d = close
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) and min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr)/ (max(high)-min(low))) / log10(14)
    range_14 = max_high - min_low
    chop = 100 * np.log10(sum_atr / range_14) / np.log10(14)
    
    # Volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian upper band
            # Only in trending market (Choppiness Index < 61.8) with volume confirmation
            if (close[i] > upper_1w_aligned[i] and 
                chop[i] < 61.8 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian lower band
            # Only in trending market with volume confirmation
            elif (close[i] < lower_1w_aligned[i] and 
                  chop[i] < 61.8 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower band or chop too high (ranging)
            if (close[i] < lower_1w_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper band or chop too high
            if (close[i] > upper_1w_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals