#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_alligator_elder_ray"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Alligator lines (Williams)
    # Jaw (Blue) - 13-period SMMA, shifted 8 bars
    jaw_raw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red) - 8-period SMMA, shifted 5 bars
    teeth_raw = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips (Green) - 5-period SMMA, shifted 3 bars
    lips_raw = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align Alligator to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter - 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray conditions: Bull Power > 0 and rising, Bear Power < 0 and falling
        elder_long = bull_power_aligned[i] > 0 and (i == 100 or bull_power_aligned[i] > bull_power_aligned[i-1])
        elder_short = bear_power_aligned[i] < 0 and (i == 100 or bear_power_aligned[i] < bear_power_aligned[i-1])
        
        # Long: Alligator uptrend + Elder Ray bullish + volume
        long_signal = alligator_long and elder_long and volume_ok[i]
        # Short: Alligator downtrend + Elder Ray bearish + volume
        short_signal = alligator_short and elder_short and volume_ok[i]
        
        # Exit when Alligator reverses (Lips crosses Teeth)
        exit_long = lips_aligned[i] < teeth_aligned[i] and position == 1
        exit_short = lips_aligned[i] > teeth_aligned[i] and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals