#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels (H3/L3) with volume confirmation and chop regime filter
# Long when price touches 1w L3 support AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range regime)
# Short when price touches 1w H3 resistance AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range regime)
# Exit when price reverts to 1w pivot (midpoint) or opposite Camarilla level touched
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Camarilla H3/L3 levels act as magnet levels in ranging markets
# Volume confirmation ensures participation on touch
# Chop regime filter (>61.8) ensures we only trade in ranging markets where mean reversion works
# Works in both bull (buy range dips) and bear (sell range rallies) markets

name = "1d_1wCamarillaH3L3_VolumeChop"
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
    
    # Get 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 completed weekly bars for pivot calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels: based on previous week's OHLC
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # Pivot = (high + low + close)/3
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # First bar will have NaN due to roll, handled by min_periods equivalent
    
    camarilla_h3 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 4
    camarilla_l3 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 4
    camarilla_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate Chopiness Index regime filter (14-period)
    # Chop > 61.8 = ranging market (good for mean reversion)
    # Chop < 38.2 = trending market (avoid for this strategy)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log((max_high_14 - min_low_14) / atr_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) <= 0, 50, chop)  # Handle division by zero
    chop_regime = chop > 61.8  # Only trade in ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches L3 support, volume confirmation, chop regime (ranging)
            if (low[i] <= camarilla_l3_aligned[i] and 
                volume_confirm[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches H3 resistance, volume confirmation, chop regime (ranging)
            elif (high[i] >= camarilla_h3_aligned[i] and 
                  volume_confirm[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to pivot or touches H3 (stop loss)
            if (high[i] >= camarilla_h3_aligned[i] or  # Stop loss if breaks above H3
                close[i] >= camarilla_pivot_aligned[i]):  # Take profit at pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to pivot or touches L3 (stop loss)
            if (low[i] <= camarilla_l3_aligned[i] or  # Stop loss if breaks below L3
                close[i] <= camarilla_pivot_aligned[i]):  # Take profit at pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals