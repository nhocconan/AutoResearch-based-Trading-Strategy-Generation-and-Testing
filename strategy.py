#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + Volume Spike + 1d Choppiness Regime Filter
# Donchian channel breakouts capture trend continuation with high probability.
# Volume spike confirms institutional participation in the breakout.
# 1d Choppiness Index > 61.8 indicates ranging market (avoid breakouts in chop).
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_VolumeSpike_1dChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (hh_14 - ll_14)) / np.log10(14)
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # avoid division by zero
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long: Close above Upper Donchian AND chop > 61.8 (trending) AND volume spike
            if close_val > high_20_val and chop_val > 61.8 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Lower Donchian AND chop > 61.8 (trending) AND volume spike
            elif close_val < low_20_val and chop_val > 61.8 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below Lower Donchian (reversal) or at Upper Donchian (take profit)
            if close_val < low_20_val or close_val >= high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above Upper Donchian (reversal) or at Lower Donchian (take profit)
            if close_val > high_20_val or close_val <= low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals