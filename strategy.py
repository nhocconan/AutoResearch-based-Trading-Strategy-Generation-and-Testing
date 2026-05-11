#!/usr/bin/env python3
"""
6h_Alligator_ElderRay_Volume_SmartMoney
Hypothesis: Combine Bill Williams Alligator (trend) with Elder Ray Index (bull/bear power) and volume spikes.
In strong trends (Alligator aligned + Elder Ray confirms direction), enter on pullbacks with volume confirmation.
Exit when Alligator reverses or Elder Ray weakens. Works in bull by buying dips in uptrends and in bear by selling rallies in downtrends.
Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag while capturing strong moves.
"""

name = "6h_Alligator_ElderRay_Volume_SmartMoney"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Elder Ray calculation (requires 13 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Get 12h data for Alligator (requires 13, 8, 5 SMAs)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Elder Ray Index ---
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # --- 12h Alligator (Bill Williams) ---
    # Jaw = SMA(13, 8), Teeth = SMA(8, 5), Lips = SMA(5, 3)
    close_12h = df_12h['close'].values
    
    # Jaw (blue line) - 13-period SMA, 8 periods ahead
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)  # shift 8 periods forward
    jaw_12h[:8] = np.nan
    
    # Teeth (red line) - 8-period SMA, 5 periods ahead
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)  # shift 5 periods forward
    teeth_12h[:5] = np.nan
    
    # Lips (green line) - 5-period SMA, 3 periods ahead
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)  # shift 3 periods forward
    lips_12h[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i])
        alligator_short = (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i])
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_power_rising = bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]
        bear_power_falling = bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]
        
        if position == 0:
            # Look for entries with trend confirmation and volume spike
            if alligator_long and bull_power_1d_aligned[i] > 0 and bull_power_rising and volume_spike[i]:
                # Long: uptrend + bull power positive + volume spike
                signals[i] = 0.25
                position = 1
            elif alligator_short and bear_power_1d_aligned[i] < 0 and bear_power_falling and volume_spike[i]:
                # Short: downtrend + bear power negative + volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Alligator turns down OR Bull Power turns negative
                if not alligator_long or bull_power_1d_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Alligator turns up OR Bear Power turns positive
                if not alligator_short or bear_power_1d_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals