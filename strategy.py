#!/usr/bin/env python3
# 1D_WilliamsAlligator_1wTrend_Volume
# Hypothesis: Uses Williams Alligator on 1d timeframe to determine trend (Jaw, Teeth, Lips alignment).
# Enters long when price is above all three lines (bullish alignment) with volume confirmation.
# Enters short when price is below all three lines (bearish alignment) with volume confirmation.
# Uses weekly EMA40 as higher timeframe trend filter to avoid counter-trend trades.
# Exits when price crosses back below/above the Teeth line or trend changes.
# Designed for 1d timeframe with position size 0.25 to target 10-25 trades per year.

name = "1D_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (SMA with specific periods)
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw_raw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    
    # Apply forward shift (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set initial values to NaN due to shift
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d (no shift needed as already on 1d)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema_40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Warmup for Alligator (max of 13,8,5 periods)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        bullish_alignment = (close[i] > jaw_aligned[i] and 
                           close[i] > teeth_aligned[i] and 
                           close[i] > lips_aligned[i])
        bearish_alignment = (close[i] < jaw_aligned[i] and 
                           close[i] < teeth_aligned[i] and 
                           close[i] < lips_aligned[i])
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_40_1w_aligned[i]
        weekly_downtrend = close[i] < ema_40_1w_aligned[i]
        
        if position == 0:
            # Long entry: bullish Alligator alignment + volume confirmation + weekly uptrend
            if (bullish_alignment and 
                volume_confirm[i] and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + volume confirmation + weekly downtrend
            elif (bearish_alignment and 
                  volume_confirm[i] and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Teeth or weekly trend turns down
            if (close[i] < teeth_aligned[i] or 
                not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Teeth or weekly trend turns up
            if (close[i] > teeth_aligned[i] or 
                not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals