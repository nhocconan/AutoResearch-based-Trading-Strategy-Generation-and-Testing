#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d chart with 1w Williams Alligator filter and 1d pivot point breakout.
# Long when price breaks above R1 pivot with price above Alligator teeth and green alignment.
# Short when price breaks below S1 pivot with price below Alligator teeth and red alignment.
# Uses weekly Alligator to filter trend direction and avoid counter-trend trades.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot points (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 1d timeframe (identity)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2
    
    # Alligator Jaw (blue line): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Alligator Teeth (red line): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Alligator Lips (green line): 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Align Alligator components to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips.values)
    
    # 1d data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ok = vol_filter[i]
        
        # Alligator alignment
        green_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        red_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Price relative to teeth
        price_above_teeth = price > teeth_val
        price_below_teeth = price < teeth_val
        
        if position == 0:
            # Long: price breaks above R1, above teeth, green alignment, volume
            if price > r1_val and price_above_teeth and green_alignment and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below teeth, red alignment, volume
            elif price < s1_val and price_below_teeth and red_alignment and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or red alignment
            if price < s1_val or red_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or green alignment
            if price > r1_val or green_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Alligator_1d_Pivot_Breakout_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0