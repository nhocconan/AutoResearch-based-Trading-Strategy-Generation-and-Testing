#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index Regime Filter with Donchian Breakout
# - Uses Choppiness Index (14) on 1d to determine market regime
# - CHOP > 61.8 = ranging (mean revert at Donchian bands)
# - CHOP < 38.2 = trending (follow Donchian breakout)
# - Donchian(20) breakout with volume confirmation
# - Works in bull/bear by adapting to regime
# - Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag on 4h timeframe

name = "4h_ChoppinessRegime_DonchianBreakout_Volume"
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
    
    # 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR(14) for 1d
    atr_14 = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Choppiness Index(14)
    chop = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        atr_sum = np.nansum(atr_14[i-13:i+1])
        if atr_sum > 0:
            high_low_range = np.max(high_1d[i-13:i+1]) - np.min(low_1d[i-13:i+1])
            chop[i] = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, donchian_len - 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Determine regime based on Choppiness Index
            if chop_val > 61.8:  # Ranging market - mean revert
                # Long: price touches lower band
                if close[i] <= lower[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches upper band
                elif close[i] >= upper[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_val < 38.2:  # Trending market - follow breakout
                # Long: breakout above upper band with volume spike
                if close[i] > upper[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakdown below lower band with volume spike
                elif close[i] < lower[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below midline or opposite band
            midline = (upper[i] + lower[i]) / 2
            if close[i] < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above midline or opposite band
            midline = (upper[i] + lower[i]) / 2
            if close[i] > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals