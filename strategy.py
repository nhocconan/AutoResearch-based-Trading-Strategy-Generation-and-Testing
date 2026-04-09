#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# In trending markets, breakouts capture momentum; in ranging markets (2025+), chop filter avoids false breakouts.
# Volume confirmation ensures breakouts have conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years by requiring Donchian breakout + volume spike + chop < 61.8 (trending) or > 61.8 (range) with appropriate logic.
# Primary timeframe: 4h, HTF: 1d for regime filter (chop) to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # 1d HTF data for chop regime (to avoid look-ahead and use completed 1d bars)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14))) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(atr_14 / (np.log10(14) * range_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h - using close-based breakout for clarity
    lookback = 20
    highest_high = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low OR volume dries up OR chop becomes too high (choppy market)
            if close[i] < lowest_low[i] or not volume_confirmed or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high OR volume dries up OR chop becomes too high
            if close[i] > highest_high[i] or not volume_confirmed or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high
                # In trending markets (chop < 38.2), breakouts are more reliable
                # In ranging markets (chop > 61.8), we avoid breakouts (mean reversion instead)
                # For simplicity: only trade breakouts when NOT in extreme chop (chop <= 61.8)
                if close[i] > highest_high[i] and chop_aligned[i] <= 61.8:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < lowest_low[i] and chop_aligned[i] <= 61.8:
                    position = -1
                    signals[i] = -0.25
    
    return signals