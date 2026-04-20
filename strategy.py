#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_Signal_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d: Williams Alligator (13,8,5 SMAs with future shift) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMMA of median price, shifted 8 bars forward)
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA of median price, shifted 5 bars forward)
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA of median price, shifted 3 bars forward)
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) with volume
            if (lips_val > teeth_val > jaw_val and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) with volume
            elif (jaw_val > teeth_val > lips_val and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips cross below Teeth (bullish momentum fading)
            if lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips cross above Teeth (bearish momentum fading)
            if lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals