#!/usr/bin/env python3
"""
6h Williams Alligator + Weekly Pivot Direction + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence on 6h.
Weekly pivot R1/S1 from 1w provides institutional bias. Long when price > weekly R1,
Alligator aligned bullish (Lips > Teeth > Jaw), and volume spike confirms.
Short when price < weekly S1, Alligator bearish (Lips < Teeth < Jaw), volume spike.
Designed for low-moderate trade frequency (12-37/year) on 6h to work in bull/bear via
trend alignment and institutional levels, avoiding chop via Alligator convergence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R1, S1) from prior week's OHLC
    # Use shift(1) to avoid look-ahead: prior week's data for current week's levels
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_open = df_1w['open'].shift(1).values
    
    # Weekly pivot: PP = (H + L + C) / 3
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # R1 = (2 * PP) - L
    weekly_r1 = (2 * weekly_pp) - prev_week_low
    # S1 = (2 * PP) - H
    weekly_s1 = (2 * weekly_pp) - prev_week_high
    
    # Align weekly R1/S1 to 6h (no extra delay - based on completed weekly bar)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Williams Alligator on 6h: SMAs of median price (HL/2)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # Jaw (Blue): 13-period SMA, 8-bar shift
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (Red): 8-period SMA, 5-bar shift
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (Green): 5-period SMA, 3-bar shift
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8 + jaw period 13 = 21), volume MA 20
    start_idx = max(21, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        r1_level = weekly_r1_aligned[i]
        s1_level = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        # Bullish: Lips > Teeth > Jaw (green > red > blue)
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish: Lips < Teeth < Jaw (green < red < blue)
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:
            # Look for entry signals
            # Long: price > weekly R1 AND Alligator bullish AND volume spike
            long_entry = (curr_close > r1_level) and bullish_alignment and vol_spike
            # Short: price < weekly S1 AND Alligator bearish AND volume spike
            short_entry = (curr_close < s1_level) and bearish_alignment and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below weekly S1 (broken support) OR Alligator turns bearish
            if (curr_close < s1_level) or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly R1 (broken resistance) OR Alligator turns bullish
            if (curr_close > r1_level) or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_WeeklyPivot_R1S1_VolumeSpike"
timeframe = "6h"
leverage = 1.0