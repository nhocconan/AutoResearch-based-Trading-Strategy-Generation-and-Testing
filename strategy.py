#!/usr/bin/env python3
# 4h_donchian_20_volume_chop_regime_v2
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Donchian breakout captures structural moves; volume > 1.5x 20-period average confirms institutional participation.
# Choppiness index (CHOP) > 61.8 = ranging market (mean reversion), CHOP < 38.2 = trending (breakout continuation).
# Only take breakout signals when CHOP < 38.2 (trending regime) to avoid false breakouts in ranging markets.
# Works in bull markets via upward breakouts and bear markets via downward breakouts.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_volume_chop_regime_v2"
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
    
    # 12h HTF data for choppiness regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = np.nan
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - prev_close_12h)
    tr3 = np.abs(low_12h - prev_close_12h)
    tr_12h = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR(14) on 12h
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM for 12h
    high_diff = np.diff(high_12h, prepend=np.nan)
    low_diff = np.diff(low_12h, prepend=np.nan)
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smoothed +DM, -DM, ATR
    atr_12h_smooth = pd.Series(atr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di_12h = 100 * plus_dm_smooth / atr_12h_smooth
    minus_di_12h = 100 * minus_dm_smooth / atr_12h_smooth
    
    # DX and Choppiness Index
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    chop_12h = 100 * np.log10(pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values / 
                              (atr_12h_smooth * 14)) / np.log10(14)
    
    # Align 12h chop to 4h
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian midpoint OR chop > 61.8 (ranging regime)
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint or chop_12h_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian midpoint OR chop > 61.8 (ranging regime)
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint or chop_12h_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trending regime (CHOP < 38.2)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            trending_regime = chop_12h_aligned[i] < 38.2
            
            if volume_confirmed and trending_regime:
                # Long: price breaks above Donchian high
                if close[i] > highest_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals