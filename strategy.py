#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Elder Ray volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order) OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams Alligator identifies trend structure with built-in smoothing. Elder Ray confirms bull/bear power with volume.
Designed to work in both bull and bear markets by requiring volume confirmation and clear Alligator alignment.
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
    
    # Load 6h data for Alligator - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 1:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams Alligator lines (Smoothed Moving Average = SMA)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    jaw_6h = np.roll(jaw_6h, 8)  # shift forward 8 bars
    jaw_6h[:8] = np.nan
    
    teeth_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    teeth_6h = np.roll(teeth_6h, 5)  # shift forward 5 bars
    teeth_6h[:5] = np.nan
    
    lips_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips_6h = np.roll(lips_6h, 3)  # shift forward 3 bars
    lips_6h[:3] = np.nan
    
    # Load 1d data for Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align HTF indicators to 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        # Check Alligator alignment
        jaw_val = jaw_6h_aligned[i]
        teeth_val = teeth_6h_aligned[i]
        lips_val = lips_6h_aligned[i]
        
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: bullish Alligator AND bullish Elder Ray AND volume spike
            if (bullish_alignment and 
                bull_power_1d_aligned[i] > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND bearish Elder Ray AND volume spike
            elif (bearish_alignment and 
                  bear_power_1d_aligned[i] < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator alignment breaks
            if position == 1 and not bullish_alignment:
                exit_signal = True
            elif position == -1 and not bearish_alignment:
                exit_signal = True
            
            # Secondary exit: volume drops below average
            if volume[i] < vol_ma_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0