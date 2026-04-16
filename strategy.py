#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily pivot point R1/S1 breakout with volume confirmation.
# Long when price breaks above daily R1 pivot with volume > 1.5x 20-period median.
# Short when price breaks below daily S1 pivot with volume > 1.5x 20-period median.
# Uses discrete position size 0.25. Exits when price returns to daily pivot (mean reversion).
# Daily pivots provide structure from higher timeframe, volume confirms breakout validity.
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Daily Indicators: Pivot Points (based on prior day) ===
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using prior day's values to avoid look-ahead
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r1_1d = 2 * pivot_1d - np.roll(low_1d, 1)
    s1_1d = 2 * pivot_1d - np.roll(high_1d, 1)
    
    # === Daily Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (12h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
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
        
        # Get current daily volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current daily volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below daily pivot (mean reversion)
            if price <= pivot_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above daily pivot (mean reversion)
            if price >= pivot_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above daily R1 with volume spike
            if price > r1 and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below daily S1 with volume spike
            elif price < s1 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_DailyPivot_R1S1_Breakout_VolumeSpike1.5x_v1"
timeframe = "12h"
leverage = 1.0