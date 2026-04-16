#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend(10,3) for trend direction + 4h Donchian(20) breakout + volume confirmation.
# Long when 12h Supertrend is bullish AND price breaks above 4h Donchian(20) upper + volume > 1.3x 20-period median volume.
# Short when 12h Supertrend is bearish AND price breaks below 4h Donchian(20) lower + volume > 1.3x 20-period median volume.
# Exit when price returns to Donchian middle (avg of upper/lower) OR Supertrend flips.
# Uses discrete position size 0.25. Targets 20-50 trades/year to minimize fee drag.
# Supertrend filters for strong trends, Donchian captures breakouts, volume confirms institutional participation.
# Works in both bull and bear markets by only trading in the direction of the 12h trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Supertrend(10,3) ===
    # True Range
    high_low_12h = high_12h - low_12h
    high_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    true_range_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    
    # ATR(10)
    atr_10 = pd.Series(true_range_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_basic = hl2_12h + (3.0 * atr_10)
    lower_basic = hl2_12h - (3.0 * atr_10)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        # Upper Band
        if upper_basic[i] < supertrend[i-1] or close_12h[i-1] > supertrend[i-1]:
            upper_band = upper_basic[i]
        else:
            upper_band = supertrend[i-1]
        
        # Lower Band
        if lower_basic[i] > supertrend[i-1] or close_12h[i-1] < supertrend[i-1]:
            lower_band = lower_basic[i]
        else:
            lower_band = supertrend[i-1]
        
        # Supertrend
        if direction[i-1] == 1 and close_12h[i] <= lower_band:
            direction[i] = -1
            supertrend[i] = upper_band
        elif direction[i-1] == -1 and close_12h[i] >= upper_band:
            direction[i] = 1
            supertrend[i] = lower_band
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = lower_band
            else:
                supertrend[i] = upper_band
    
    # Supertrend direction (1 = uptrend, -1 = downtrend)
    supertrend_direction = direction
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    donchian_middle = (highest_20 + lowest_20) / 2.0
    
    # === 4h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_direction)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20)  # Donchian20 needs 20, volume median needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        st_dir = supertrend_dir_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.3x median volume
        volume_spike = volume[i] > (vol_median * 1.3)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle OR Supertrend turns bearish
            if (price <= middle) or (st_dir == -1):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle OR Supertrend turns bullish
            if (price >= middle) or (st_dir == 1):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Supertrend bullish AND price breaks above upper Donchian + volume spike
            if (st_dir == 1) and (price > upper) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Supertrend bearish AND price breaks below lower Donchian + volume spike
            elif (st_dir == -1) and (price < lower) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_12hSupertrend10_3_Donchian20_VolumeSpike1.3x_V1"
timeframe = "4h"
leverage = 1.0