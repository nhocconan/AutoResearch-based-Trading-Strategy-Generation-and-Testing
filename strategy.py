#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly volume confirmation.
# Uses Donchian channel for breakout direction, 1d ATR to filter low volatility regimes,
# and weekly volume surge to confirm institutional interest. Works in bull/bear via
# volatility filter and volume confirmation, targeting 15-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # Align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_surge_1w = volume_1w > 1.5 * vol_ma20_1w  # Weekly volume surge
    
    # Align 1d ATR and weekly volume surge to 6h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_surge_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_surge_1w)
    
    # Calculate 6h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(vol_surge_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian upper + sufficient volatility (ATR > 0) + weekly volume surge
            if (close[i] > donchian_upper[i] and 
                atr_14_aligned[i] > 0 and 
                vol_surge_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian lower + sufficient volatility + weekly volume surge
            elif (close[i] < donchian_lower[i] and 
                  atr_14_aligned[i] > 0 and 
                  vol_surge_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite Donchian touch or volatility collapse
            if position == 1:
                if close[i] < donchian_lower[i] or atr_14_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_upper[i] or atr_14_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_VolSurge1w"
timeframe = "6h"
leverage = 1.0