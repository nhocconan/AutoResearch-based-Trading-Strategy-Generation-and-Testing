#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R1/S1, R2/S2, R3/S3, R4/S4) with 6h volume confirmation.
# Long when price breaks above R1 with volume > 1.5x 20-period median volume and closes above R1.
# Short when price breaks below S1 with volume > 1.5x 20-period median volume and closes below S1.
# Exit when price returns to the weekly pivot point (PP) or opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses weekly Camarilla levels derived from prior week's OHLC to avoid look-ahead.
# Volume filter reduces false breakouts. Discrete position size 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).
# Weekly pivots adapt to changing volatility and work in both bull and bear markets by capturing institutional order flow
# around key weekly support/resistance levels, with volume confirmation ensuring genuine participation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for volume median
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    vol_6h = df_6h['volume'].values
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    range_1w = high_1w - low_1w
    
    # Camarilla levels (based on prior week)
    r1_1w = pp_1w + range_1w * 1.1 / 12
    s1_1w = pp_1w - range_1w * 1.1 / 12
    r2_1w = pp_1w + range_1w * 1.1 / 6
    s2_1w = pp_1w - range_1w * 1.1 / 6
    r3_1w = pp_1w + range_1w * 1.1 / 4
    s3_1w = pp_1w - range_1w * 1.1 / 4
    r4_1w = pp_1w + range_1w * 1.1 / 2
    s4_1w = pp_1w - range_1w * 1.1 / 2
    
    # Align all weekly levels to 6h timeframe (completed weekly bars only)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # volume median(20) + weekly data buffer
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_6h = vol_6h_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Weekly levels (already aligned)
        pp = pp_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to weekly pivot point (PP) or breaks below S1
            if price <= pp or price <= s1:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to weekly pivot point (PP) or breaks above R1
            if price >= pp or price >= r1:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.5x median volume
            volume_spike = vol_6h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above R1 with volume spike and closes above R1
            if price > r1 and close[i] > r1 and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below S1 with volume spike and closes below S1
            elif price < s1 and close[i] < s1 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WeeklyCamarilla_R1S1_Breakout_VolumeSpike1.5x_v1"
timeframe = "6h"
leverage = 1.0