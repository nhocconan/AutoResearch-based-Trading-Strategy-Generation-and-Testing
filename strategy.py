#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Long when price breaks above 20-period 12h Donchian high AND 1d volume > 1.5x 20-period average AND 1d choppiness > 61.8 (range)
# Short when price breaks below 20-period 12h Donchian low AND 1d volume > 1.5x 20-period average AND 1d choppiness > 61.8 (range)
# Exit when price returns to the 12-period Donchian midpoint OR choppiness < 38.2 (trending)
# Uses discrete sizing 0.25 to minimize fee churn. Works in ranging markets (chop > 61.8) where reversals are reliable.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL

name = "12h_Donchian20_VolumeSpike_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume 20-period EMA
    volume_1d_series = pd.Series(volume_1d)
    volume_ema20_1d = volume_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ratio_1d = volume_1d / volume_ema20_1d  # Current volume / 20-period average
    
    # Calculate 1d choppiness index
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: 100 * log10(sum(tr14) / (max_high14 - min_low14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero or near-zero
    safe_range = np.where(range_14 < 1e-10, 1e-10, range_14)
    choppiness = 100 * np.log10(tr_sum_14 / safe_range) / np.log10(14)
    choppiness = np.where(np.isnan(choppiness) | np.isinf(choppiness), 50.0, choppiness)  # Neutral if invalid
    
    # Align 1d indicators to 12h
    volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    # Calculate 12h Donchian channels (20-period)
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (max_high_20 + min_low_20) / 2.0  # Midpoint for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any value is NaN
        if (np.isnan(volume_ratio_1d_aligned[i]) or np.isnan(choppiness_aligned[i]) or 
            np.isnan(max_high_20[i]) or np.isnan(min_low_20[i]) or np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        in_range = choppiness_aligned[i] > 61.8
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + in range
            if close[i] > max_high_20[i] and volume_spike and in_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + in range
            elif close[i] < min_low_20[i] and volume_spike and in_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR choppiness < 38.2 (trending)
            if close[i] <= donchian_mid[i] or choppiness_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR choppiness < 38.2 (trending)
            if close[i] >= donchian_mid[i] or choppiness_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals