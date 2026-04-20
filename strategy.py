#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_Signal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d: Williams Alligator components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period smoothed, 8-period offset)
    median_1d = (high_1d + low_1d) / 2.0
    jaw_raw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # 8-period forward shift (offset)
    jaw_values = jaw.values
    
    # Teeth (8-period smoothed, 5-period offset)
    teeth_raw = pd.Series(median_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # 5-period forward shift
    teeth_values = teeth.values
    
    # Lips (5-period smoothed, 3-period offset)
    lips_raw = pd.Series(median_1d).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)  # 3-period forward shift
    lips_values = lips.values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_values)
    
    # === 6h: Price action and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment (Lips > Teeth > Jaw) + price above all + volume
            if (lips_val > teeth_val and teeth_val > jaw_val and  # Bullish alignment
                close_val > lips_val and  # Price above lips
                vol_ratio_val > 1.3):     # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Lips < Teeth < Jaw) + price below all + volume
            elif (lips_val < teeth_val and teeth_val < jaw_val and  # Bearish alignment
                  close_val < lips_val and  # Price below lips
                  vol_ratio_val > 1.3):     # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Teeth or alignment turns bearish
            if (close_val < teeth_val or 
                lips_val < teeth_val or teeth_val < jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Teeth or alignment turns bullish
            if (close_val > teeth_val or 
                lips_val > teeth_val or teeth_val > jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals