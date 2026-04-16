#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) upper + volume > 1.5x 20-period median volume + CHOP(14) > 61.8 (ranging market).
# Short when price breaks below Donchian(20) lower + volume > 1.5x 20-period median volume + CHOP(14) > 61.8.
# Uses discrete position size 0.25. Exits when price returns to Donchian middle (avg of upper/lower) or when chop regime shifts to trending (CHOP < 38.2).
# Donchian breakouts capture volatility expansion. Volume confirmation ensures institutional participation.
# Chop regime filter ensures we only trade in ranging markets where mean reversion at channel extremes works best.
# 12h timeframe targets 12-37 trades/year to minimize fee drag. Works in both bull and bear markets by fading extremes in ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    donchian_middle = (highest_20 + lowest_20) / 2.0
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # === 1d Indicators: Choppiness Index (14) ===
    # True Range
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # Sum of True Range over 14 periods
    atr_sum_14 = pd.Series(true_range_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: CHOP = 100 * log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop_14 = np.where(
        range_14 > 0,
        100 * np.log10(atr_sum_14 / range_14) / np.log10(14),
        50  # neutral when range is zero
    )
    
    # Regime filters: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_ranging = chop_14 > 61.8
    chop_trending = chop_14 < 38.2
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 14)  # Donchian20 needs 20, volume median needs 20, chop needs 14
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or np.isnan(chop_trending_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        chop_ranging_val = chop_ranging_aligned[i]
        chop_trending_val = chop_trending_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle OR chop regime shifts to trending
            if (price <= middle) or chop_trending_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle OR chop regime shifts to trending
            if (price >= middle) or chop_trending_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + ranging market
            if (price > upper) and volume_spike and chop_ranging_val:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower Donchian + volume spike + ranging market
            elif (price < lower) and volume_spike and chop_ranging_val:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_VolumeSpike1.5x_ChopRangingFilter_V1"
timeframe = "12h"
leverage = 1.0