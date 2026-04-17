#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike
Long: Jaw < Teeth < Lips (bullish alignment) + volume > 1.5x 20-period volume SMA
Short: Jaw > Teeth > Lips (bearish alignment) + volume > 1.5x 20-period volume SMA
Exit: Loss of Alligator alignment (jaws cross teeth or lips)
Williams Alligator uses smoothed moving averages (SMMA) with specific periods.
Williams Alligator is effective in both trending and ranging markets when combined with volume confirmation.
Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (13, 8, 5 periods with 8, 5, 3 shifts)
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply shifts: Jaw 8 bars, Teeth 5 bars, Lips 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set invalid values for shifted portions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 13)  # need volume SMA and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_aligned = jaw_val < teeth_val < lips_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_aligned = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: Bullish alignment + volume spike
            if bullish_aligned and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike
            elif bearish_aligned and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Loss of bullish alignment
            if not bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Loss of bearish alignment
            if not bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_VolumeSpike"
timeframe = "1d"
leverage = 1.0