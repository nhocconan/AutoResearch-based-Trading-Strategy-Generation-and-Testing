#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot point R1/S1 breakout with volume confirmation.
# Long when price breaks above weekly R1 pivot with volume > 1.5x 20-period median.
# Short when price breaks below weekly S1 pivot with volume > 1.5x 20-period median.
# Uses discrete position size 0.25. No trailing stop - relies on mean reversion at opposite pivot level.
# Weekly pivots provide structure from higher timeframe, volume confirms breakout validity.
# 6h timeframe balances trade frequency and noise reduction for BTC/ETH in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === Weekly Indicators: Pivot Points (based on prior week) ===
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using prior week's values to avoid look-ahead
    pivot_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3
    r1_1w = 2 * pivot_1w - np.roll(low_1w, 1)
    s1_1w = 2 * pivot_1w - np.roll(high_1w, 1)
    
    # === Weekly Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    vol_median_aligned = align_htf_to_ltf(prices, df_1w, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Volume median needs 20 periods
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol_median = vol_median_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Get current weekly volume for volume spike filter
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        current_vol_1w = vol_1w_aligned[i]
        
        # Volume spike filter: current weekly volume > 1.5x median volume
        volume_spike = current_vol_1w > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below weekly pivot (mean reversion)
            if price <= pivot_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above weekly pivot (mean reversion)
            if price >= pivot_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above weekly R1 with volume spike
            if price > r1 and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below weekly S1 with volume spike
            elif price < s1 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_VolumeSpike1.5x_v1"
timeframe = "6h"
leverage = 1.0