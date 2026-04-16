#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d pivot points (R1/S1) with volume confirmation and ATR filter.
# Long when price breaks above R1 with volume > 1.3x 20-period median volume AND ATR(14) > 0.5 * ATR(50) (volatility expansion).
# Short when price breaks below S1 with volume > 1.3x 20-period median volume AND ATR(14) > 0.5 * ATR(50).
# Uses discrete position size 0.25. Exits on opposite pivot break (price < S1 for longs, price > R1 for shorts) or ATR contraction (ATR(14) < 0.3 * ATR(50)).
# Pivot points provide institutional reference levels; volume confirmation ensures participation; ATR filter avoids low-volatility false breakouts.
# 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Daily pivots are more stable than weekly for 6h trading and avoid look-ahead with proper alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Pivot Points (Standard) ===
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L
    # S1 = 2*Pivot - H
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # === 1d Indicators: ATR (14-period and 50-period for expansion/contraction filter) ===
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_14_1d = pd.Series(true_range_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(true_range_1d).rolling(window=50, min_periods=50).mean().values
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 50, 20)  # ATR50 needs 50, ATR14 needs 14, volume median needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        atr_14 = atr_14_aligned[i]
        atr_50 = atr_50_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.3x median volume
        volume_spike = current_vol_1d > (vol_median * 1.3)
        
        # ATR expansion filter: ATR(14) > 0.5 * ATR(50) (volatility expansion)
        atr_expansion = atr_14 > (atr_50 * 0.5)
        
        # ATR contraction filter: ATR(14) < 0.3 * ATR(50) (low volatility)
        atr_contraction = atr_14 < (atr_50 * 0.3)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < S1 (opposite pivot break) OR ATR contraction
            if (price < s1_aligned[i]) or atr_contraction:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > R1 (opposite pivot break) OR ATR contraction
            if (price > r1_aligned[i]) or atr_contraction:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > R1 + volume spike + ATR expansion
            if (price > r1_aligned[i]) and volume_spike and atr_expansion:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < S1 + volume spike + ATR expansion
            elif (price < s1_aligned[i]) and volume_spike and atr_expansion:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dPivotR1S1_Breakout_VolumeSpike1.3x_ATRexpansion0.5x_EXIToppositePivot_ATRcontraction0.3x_v1"
timeframe = "6h"
leverage = 1.0