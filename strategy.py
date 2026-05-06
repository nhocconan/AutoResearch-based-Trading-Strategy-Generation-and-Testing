#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and 1w Choppiness regime filter
# - Uses 1d Donchian(20) breakout for entry signals
# - Uses 1w Choppiness Index to filter range (CHOP > 61.8) vs trend (CHOP < 38.2) regimes
# - Enters long when price breaks above 1d Donchian upper in trending regime with volume spike
# - Enters short when price breaks below 1d Donchian lower in trending regime with volume spike
# - Exits when price returns to 1d Donchian midpoint or regime changes to range
# - Designed to capture breakouts in trending markets while avoiding false signals in ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dDonchian_1wChop_Volume_Breakout"
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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Donchian Channel (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1w Choppiness Index (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = np.nancumsum(tr) - np.nancumsum(np.where(np.arange(len(tr)) < 14, 0, np.roll(tr, 14)))
    sum_tr_14[:13] = 0  # Not enough data for first 13 periods
    for i in range(13, len(tr)):
        sum_tr_14[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_12h = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Align 1w Choppiness Index to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume filters (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or np.isnan(donchian_middle_12h[i]) or
            np.isnan(chop_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for trending regime (CHOP < 38.2) and volume spike
            trending_regime = chop_12h[i] < 38.2
            
            if trending_regime:
                # Long: price breaks above 1d Donchian upper with volume spike
                if close[i] > donchian_upper_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d Donchian lower with volume spike
                elif close[i] < donchian_lower_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR regime changes to range (CHOP > 61.8)
            if close[i] < donchian_middle_12h[i] or chop_12h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR regime changes to range (CHOP > 61.8)
            if close[i] > donchian_middle_12h[i] or chop_12h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals