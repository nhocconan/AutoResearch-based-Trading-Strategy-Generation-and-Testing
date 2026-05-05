#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when: Price breaks above Donchian upper channel (20) AND 1d volume > 1.5x 20-period average AND choppiness < 38.2 (trending regime)
# Short when: Price breaks below Donchian lower channel (20) AND 1d volume > 1.5x 20-period average AND choppiness < 38.2 (trending regime)
# Exit when price returns to Donchian middle (mean of 20-period high/low)
# Donchian breakouts capture sustained momentum after consolidation
# Volume spike confirms institutional participation
# Choppiness filter ensures we only trade in trending markets, avoiding whipsaws in ranges
# Works in both bull and bear markets by trading breakouts in direction of the prevailing trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_DonchianBreakout_Volume_Chop"
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
    
    # Get 1d data ONCE before loop for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d choppiness index (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(tr_sum / hl_range) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels (20-period) on 12h
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Filters: volume spike (>1.5x average) and trending regime (chop < 38.2)
        volume_spike = volume[i] > (1.5 * vol_ma_20_aligned[i])
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: Break above upper Donchian with filters
            if close[i] > dc_upper[i] and volume_spike and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with filters
            elif close[i] < dc_lower[i] and volume_spike and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian (mean reversion)
            if close[i] < dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian (mean reversion)
            if close[i] > dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals