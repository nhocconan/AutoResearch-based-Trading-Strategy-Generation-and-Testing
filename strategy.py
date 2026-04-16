#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian(20) breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Donchian upper(20) AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Donchian lower(20) AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Exit when price crosses Donchian middle (10-period average of upper/lower) or chop > 61.8 (range).
# Uses discrete position size 0.25. Donchian provides structure, volume confirms conviction, chop filter avoids whipsaws in ranging markets.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (capture breakouts) and bear markets (catch breakdowns with short signals).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 12h Indicators: Donchian(20) ===
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    # Middle = (upper + lower) / 2
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_12h = (upper_12h + lower_12h) / 2.0
    
    # === 1d Indicators: Volume average and Chop filter ===
    # Volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Chop filter: EHLERS CHOPPINESS INDEX
    # Chop = 100 * log10(sum(atr,14) / (max(high,14) - min(low,14))) / log10(14)
    atr_1d = np.zeros_like(close_1d)
    atr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align all indicators to primary timeframe (12h)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=0)  # Chop is contemporaneous
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma = vol_ma_aligned[i]
        chop = chop_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # 1d volume at this point (need to align 1d volume to 12h)
        # Get 1d volume aligned
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < middle (breakdown) OR chop > 61.8 (range)
            if (price < middle) or (chop > 61.8):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > middle (breakout) OR chop > 61.8 (range)
            if (price > middle) or (chop > 61.8):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > upper AND 1d volume > 1.5x 20-period avg AND chop < 61.8 (trending)
            if (price > upper) and (vol_1d_current > 1.5 * vol_ma) and (chop < 61.8):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < lower AND 1d volume > 1.5x 20-period avg AND chop < 61.8 (trending)
            elif (price < lower) and (vol_1d_current > 1.5 * vol_ma) and (chop < 61.8):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_Breakout_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0