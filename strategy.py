#!/usr/bin/env python3
# 4h_donchian_volume_regime_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# In trending markets (CHOP < 38.2), breakouts of Donchian(20) capture momentum.
# In ranging markets (CHOP > 61.8), fade Donchian touches for mean reversion.
# Volume confirmation ensures breakouts/mean reversions have participation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate choppiness index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_1d - ll_1d
    chop_raw = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filters
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price touches opposite Donchian band or volume dries up
            if close[i] <= lowest_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches opposite Donchian band or volume dries up
            if close[i] >= highest_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if is_trending:
                    # Trend following: breakout entries
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                elif is_ranging:
                    # Mean reversion: fade Donchian touches
                    if close[i] <= lowest_low[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= highest_high[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals