#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with Elder Ray power and volume confirmation.
Long when green line > red line (bullish alignment) + Bull Power > 0 + volume spike.
Short when red line > green line (bearish alignment) + Bear Power > 0 + volume spike.
Exit when lines re-cross or power turns negative. Uses 1-day EMA13 for trend filter to avoid counter-trend trades.
Designed to capture trends with low frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-day trend direction.
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
    
    # Load 1-day data for Williams Alligator and Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines (13, 8, 5 period SMAs shifted)
    # Jaw (blue): 13-period SMMA shifted 8 bars
    # Teeth (red): 8-period SMMA shifted 5 bars
    # Lips (green): 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (Williams uses SMMA but SMA is acceptable)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift the lines as per Alligator definition
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Fill NaN from roll with first valid value
    for i in range(len(jaw_1d)):
        if np.isnan(jaw_1d[i]):
            jaw_1d[i] = jaw_1d[i+1] if i+1 < len(jaw_1d) else 0
        if np.isnan(teeth_1d[i]):
            teeth_1d[i] = teeth_1d[i+1] if i+1 < len(teeth_1d) else 0
        if np.isnan(lips_1d[i]):
            lips_1d[i] = lips_1d[i+1] if i+1 < len(lips_1d) else 0
    
    # Elder Ray Power
    # Bull Power = High - EMA13
    # Bear Power = EMA13 - Low
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align to 4h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Lips > Teeth (green > red) + Bull Power > 0 + volume spike
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Teeth > Lips (red > green) + Bear Power > 0 + volume spike
            elif (teeth_1d_aligned[i] > lips_1d_aligned[i] and 
                  bear_power_1d_aligned[i] > 0 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Lines re-cross or power turns negative
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips <= Teeth or Bull Power <= 0
                if lips_1d_aligned[i] <= teeth_1d_aligned[i] or bull_power_1d_aligned[i] <= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Teeth <= Lips or Bear Power <= 0
                if teeth_1d_aligned[i] <= lips_1d_aligned[i] or bear_power_1d_aligned[i] <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0