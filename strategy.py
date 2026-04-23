#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for trend filter
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs smoothed
- Long: Lips > Teeth > Jaw (bullish alignment) + price > Jaw + volume > 1.5x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) + price < Jaw + volume > 1.5x 20-period avg
- Exit: Price crosses back below/above Teeth line
- Uses 1d EMA50 as additional trend filter to avoid counter-trend trades
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (Alligator alignment + price above Jaw) and bear markets (reverse alignment)
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
    
    # Volume confirmation: > 1.5x 20-period average (volume filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Alligator from 1d data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines (all based on close, different periods and shifts)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_13, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_8, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMA, shifted 3 bars
    lips_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_5, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align to 12h timeframe (values from previous 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5, 50)  # Need 20 for volume MA, max periods for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator alignment conditions
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: Bullish alignment + price > Jaw + volume spike + price > 1d EMA50 (uptrend)
            if (bullish_alignment and 
                close[i] > jaw_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < Jaw + volume spike + price < 1d EMA50 (downtrend)
            elif (bearish_alignment and 
                  close[i] < jaw_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses back below Teeth line
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses back above Teeth line
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Breakout_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0