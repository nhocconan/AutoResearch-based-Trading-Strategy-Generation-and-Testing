#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above Donchian(20) upper band with volume > 1.5x 20-period average and chop > 61.8 (range regime).
# Short: Price breaks below Donchian(20) lower band with volume > 1.5x 20-period average and chop > 61.8.
# Exit: Price returns to Donchian midpoint for both long and short.
# Uses 1d timeframe for choppiness regime filter to avoid noise on lower timeframe.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
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
    
    # Donchian channel (20-period) for breakout
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate choppiness index (14-period) on 1d
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR = high-low
    
    # Sum of TR over 14 periods
    tr_s = pd.Series(tr)
    tr_sum_14 = tr_s.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop = np.full_like(tr_sum_14, 50.0, dtype=float)  # default to neutral
    mask = (hh_ll_diff > 0) & (~np.isnan(tr_sum_14)) & (~np.isnan(hh_ll_diff))
    chop[mask] = 100 * np.log10(tr_sum_14[mask] / hh_ll_diff[mask]) / np.log10(14)
    
    # Align choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion at extremes)
        regime_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume confirmation and regime filter
            if (close[i] > donchian_high[i] and volume_confirmed and regime_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume confirmation and regime filter
            elif (close[i] < donchian_low[i] and volume_confirmed and regime_filter):
                position = -1
                signals[i] = -0.25
    
    return signals