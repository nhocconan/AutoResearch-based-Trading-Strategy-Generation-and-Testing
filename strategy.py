#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1d choppiness regime filter
# Long when: price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period MA AND 1d choppiness > 61.8 (range regime)
# Short when: price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period MA AND 1d choppiness > 61.8 (range regime)
# Exit when: price returns to Donchian(20) midpoint OR choppiness < 38.2 (trend regime)
# Uses Donchian for structure, volume for conviction, chop regime to avoid whipsaws in strong trends
# Timeframe: 4h, HTF: 1d for volume and chop filter. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dVolume_Chop_Range"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 4h
    if len(high) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate 1d volume confirmation
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(len(df_1d), dtype=bool)
    
    # Calculate 1d choppiness index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Sum of TR over 14 periods
        tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
        # Avoid division by zero
        range_14 = hh_14 - ll_14
        chop_raw = np.zeros_like(tr_sum_14)
        mask = (range_14 > 0) & (~np.isnan(tr_sum_14)) & (~np.isnan(range_14))
        chop_raw[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
        chop = chop_raw
    else:
        chop = np.full(len(df_1d), np.nan)
    
    # Choppiness regime: > 61.8 = range (good for mean reversion/breakouts in range), < 38.2 = trend
    chop_range = chop > 61.8  # range regime
    chop_trend = chop < 38.2  # trend regime
    
    # Align 1d indicators to 4h timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or
            np.isnan(volume_filter_aligned[i]) or np.isnan(chop_range_aligned[i]) or 
            np.isnan(chop_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above Donchian high + volume filter + range regime
            if (close[i] > donch_high[i] and 
                volume_filter_aligned[i] == 1.0 and 
                chop_range_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below Donchian low + volume filter + range regime
            elif (close[i] < donch_low[i] and 
                  volume_filter_aligned[i] == 1.0 and 
                  chop_range_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR trend regime begins
            if (close[i] >= donch_mid[i] or chop_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR trend regime begins
            if (close[i] <= donch_mid[i] or chop_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals