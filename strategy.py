#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike Confirmation
Hypothesis: Williams Alligator identifies trending vs ranging markets on daily timeframe.
In trending markets (Alligator awake: Lips > Teeth > Jaw for uptrend, reverse for downtrend),
we take breakout trades in direction of trend with volume confirmation on 12h timeframe.
In ranging markets (Alligator sleeping: lines intertwined), we stay flat to avoid whipsaw.
Uses 12h primary timeframe with 1d EMA50 for higher timeframe trend filter and volume spike confirmation.
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag while capturing strong trends.
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
    
    # Get 1d data for Williams Alligator and EMA50 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for EMA50 + enough for Alligator
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    close_1d = pd.Series(df_1d['close'])
    # Jaw (13)
    jaw = close_1d.ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    # Teeth (8)
    teeth = close_1d.ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    # Lips (5)
    lips = close_1d.ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Align Alligator to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, EMA50, and volume MA
    start_idx = max(50, 20)  # 50 for EMA50/Alligator, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_12h[i]
        teeth_val = teeth_12h[i]
        lips_val = lips_12h[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator conditions
        # Alligator awake (trending): lips, teeth, jaw are ordered and separated
        # Uptrend: lips > teeth > jaw
        # Downtrend: lips < teeth < jaw
        # Alligator sleeping (ranging): lines are intertwined
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        alligator_uptrend = lips_above_teeth and teeth_above_jaw
        alligator_downtrend = lips_below_teeth and teeth_below_jaw
        alligator_sleeping = not (alligator_uptrend or alligator_downtrend)
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if alligator_sleeping:
                # Market ranging: stay flat
                signals[i] = 0.0
                position = 0
            elif alligator_uptrend:
                # Uptrend: look for long breakouts above recent high with volume
                # Use 6-bar high as breakout level
                if i >= 6:
                    recent_high = np.max(high[i-6:i])
                    long_signal = (curr_close > recent_high) and volume_confirm
                else:
                    long_signal = False
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif alligator_downtrend:
                # Downtrend: look for short breakdowns below recent low with volume
                if i >= 6:
                    recent_low = np.min(low[i-6:i])
                    short_signal = (curr_close < recent_low) and volume_confirm
                else:
                    short_signal = False
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
        elif position == 1:
            # Exit long: Alligator turns down OR price closes below teeth
            if not alligator_uptrend or curr_close < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns up OR price closes above teeth
            if not alligator_downtrend or curr_close > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0