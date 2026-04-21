#0:00
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === Williams Alligator indicators from 12h ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(close_12h).rolling(window=13, center=False).mean().shift(8).values
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(close_12h).rolling(window=8, center=False).mean().shift(5).values
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close_12h).rolling(window=5, center=False).mean().shift(3).values
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === Volume confirmation on 6h ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for Alligator shifts
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # Long: Lips above Teeth above Jaw (bullish alignment) + volume
            if (lips_val > teeth_val > jaw_val and
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) + volume
            elif (lips_val < teeth_val < jaw_val and
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Alligator lines cross (trend change signal)
            if position == 1 and lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Alligator_Alignment_Volume"
timeframe = "6h"
leverage = 1.0