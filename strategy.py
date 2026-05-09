#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMoothed Moving Average (SMA with period=3, then shift)
    # Jaw (Blue): 13-period SMMA shifted by 8 bars
    # Teeth (Red): 8-period SMMA shifted by 5 bars
    # Lips (Green): 5-period SMMA shifted by 3 bars
    close_1d = df_1d['close'].values
    
    # SMMA calculation: SMA then shift
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    lips = np.roll(sma5, 3)   # shifted by 3
    teeth = np.roll(sma8, 5)  # shifted by 5
    jaw = np.roll(sma13, 8)   # shifted by 8
    
    # Align to 6h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Need enough data for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(lips_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + volume spike
            if lips_val > teeth_val > jaw_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + volume spike
            elif lips_val < teeth_val < jaw_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips < Teeth (bullish alignment broken) or volume drops
            if lips_val < teeth_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips > Teeth (bearish alignment broken) or volume drops
            if lips_val > teeth_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals