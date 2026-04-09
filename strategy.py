#!/usr/bin/env python3
# 4h_donchian_volume_chop_v3
# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms institutional interest,
# chop filter avoids whipsaws in ranging markets. Target: 25-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v3"
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
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for ATR(14) and highest/lowest
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Chop regime: Choppiness Index(14) on 1d
    # CHOP = 100 * LOG10(SUM(ATR(14)) / LOG10(HIGHest HIGH - LOWest LOW) / LOG10(14)
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index 0
    
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_raw = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / chop_denom_safe) / np.log10(14)
    chop_1d = chop_raw  # already calculated with min_periods via rolling sum
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian midpoint OR chop > 61.8 (range)
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian midpoint OR chop > 61.8 (range)
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop < 61.8 (avoid strong trends for breakouts)
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            chop_filter = chop_aligned[i] < 61.8  # prefer ranging/weak trend for breakout
            
            if volume_confirmed and chop_filter:
                # Long: price breaks above upper Donchian
                if close[i] > highest_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian
                elif close[i] < lowest_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals