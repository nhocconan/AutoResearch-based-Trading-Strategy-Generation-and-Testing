#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week ATR-based volatility expansion and 1-day Donchian breakout.
# Long when price breaks above 1-day Donchian high (20) AND weekly ATR expansion > 1.3x (volatility surge).
# Short when price breaks below 1-day Donchian low (20) AND weekly ATR expansion > 1.3x.
# Exit when price returns to 1-day Donchian middle or ATR contraction < 0.8x.
# Uses volatility expansion to capture momentum bursts in both bull and bear markets,
# and Donchian channels for clear breakout levels. Designed for low trade frequency
# to minimize fee drag while capturing strong moves.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20) on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Load 1w data ONCE for ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ATR(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR (14) on 1w
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / average ATR (50-period)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20, 50)  # Need Donchian and ATR periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: ATR ratio > 1.3 indicates volatility surge
        vol_expansion = atr_ratio_aligned[i] > 1.3
        
        # Volatility contraction filter: ATR ratio < 0.8 indicates volatility collapse
        vol_contraction = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Look for Donchian breakouts with volatility expansion
            # Long: price breaks above Donchian high AND volatility expansion
            if (close[i] > donch_high_aligned[i] and 
                vol_expansion):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND volatility expansion
            elif (close[i] < donch_low_aligned[i] and 
                  vol_expansion):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or volatility contraction
            if (close[i] <= donch_mid_aligned[i] or 
                vol_contraction):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or volatility contraction
            if (close[i] >= donch_mid_aligned[i] or 
                vol_contraction):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_1w_ATR_Expansion_v1"
timeframe = "12h"
leverage = 1.0