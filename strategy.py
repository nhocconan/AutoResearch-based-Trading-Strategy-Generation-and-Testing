#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator with 1-week EMA(50) trend filter and volume spike confirmation.
Trades when all three Alligator lines (Jaw, Teeth, Lips) are aligned in the direction of the weekly trend,
with volume > 2x the 20-period average. Uses fixed position size of 0.25 to limit risk.
Designed for low-frequency trading (12-37 trades/year) to minimize fee drag and work in both bull and bear markets.
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
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Smoothed Median Price (SMP) = (High + Low) / 2
    smp = (high_12h + low_12h) / 2
    
    # Alligator lines: SMMA of SMP with different periods and shifts
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(smp).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(smp).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (Green): 5-period SMMA, shifted 3 bars
    lips = pd.Series(smp).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_vals)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Lips > Teeth > Jaw (bullish alignment) and price above weekly EMA (uptrend)
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) and price below weekly EMA (downtrend)
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips cross below Jaw or price closes below weekly EMA
                if lips_aligned[i] < jaw_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips cross above Jaw or price closes above weekly EMA
                if lips_aligned[i] > jaw_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0