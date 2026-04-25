#!/usr/bin/env python3
"""
6h Williams Alligator + 12h Weekly Pivot R1S1 Breakout with Volume Confirmation
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence on 6h.
When alligator is "sleeping" (JAW > TEETH > LIPS for downtrend, JAW < TEETH < LIPS for uptrend),
we wait for price to break weekly Camarilla R1/S1 levels with volume confirmation.
In trending markets (alligator awakened), we follow the alligator direction.
Weekly pivots provide institutional support/resistance. Designed for low trade frequency 
(12-37/year) on 6h timeframe to work in both bull and bear markets via trend following 
and mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for weekly pivot calculation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots on 12h data (approx weekly from 12h bars)
    # Use previous 12h bar's high/low/close to avoid look-ahead
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    # Weekly Camarilla R1/S1: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    weekly_r1 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12
    weekly_s1 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12
    
    # Align to LTF (6h)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_12h, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_12h, weekly_s1)
    
    # Calculate Williams Alligator on 6h close: SMAs of median price
    median_price = (high + low) / 2
    # JAW: 13-period SMMA shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for alligator, volume MA, and to avoid NaN from shift
    start_idx = max(21, 20) + 1  # 13+8 shift = 21, 20 for vol MA
    
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
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        r1_level = weekly_r1_aligned[i]
        s1_level = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator conditions
        # Sleeping downtrend: JAW > TEETH > LIPS
        sleeping_down = jaw_val > teeth_val and teeth_val > lips_val
        # Sleeping uptrend: JAW < TEETH < LIPS
        sleeping_up = jaw_val < teeth_val and teeth_val < lips_val
        # Awakened bullish: LIPS > TEETH > JAW
        awakened_bull = lips_val > teeth_val and teeth_val > jaw_val
        # Awakened bearish: LIPS < TEETH < JAW
        awakened_bear = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above weekly R1 with volume spike AND (sleeping uptrend OR awakened bullish)
            long_entry = (curr_close > r1_level) and vol_spike and (sleeping_up or awakened_bull)
            # Short: price breaks below weekly S1 with volume spike AND (sleeping downtrend OR awakened bearish)
            short_entry = (curr_close < s1_level) and vol_spike and (sleeping_down or awakened_bear)
            
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
            # Exit: price crosses below weekly S1 OR alligator turns bearish (LIPS < TEETH)
            if (curr_close < s1_level) or (lips_val < teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly R1 OR alligator turns bullish (LIPS > TEETH)
            if (curr_close > r1_level) or (lips_val > teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_WeeklyPivot_R1S1_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0