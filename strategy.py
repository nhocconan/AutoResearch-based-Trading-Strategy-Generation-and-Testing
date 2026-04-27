# Hypothesis: 4h Williams Alligator + Volume Spike + Regime Filter
# Williams Alligator uses 3 smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# In trending markets, the lines are well-separated and aligned (Jaw below Teeth below Lips for uptrend).
# In ranging markets, the lines intertwine. We use this as a trend filter.
# Volume spike confirms institutional interest. We take trades in the direction of the Alligator trend
# when volume exceeds 1.5x its 20-period average.
# This strategy avoids overtrading by requiring both trend alignment and volume confirmation.
# Works in bull/bear: Alligator identifies trend direction, volume confirms strength.
# Risk: 25% position size, exit when Alligator lines re-intertwine (trend weakens).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(source) < period:
        return np.full(len(source), np.nan)
    result = np.full(len(source), np.nan)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator (smoothed MAs)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Williams Alligator parameters (13, 8, 5) with future shifts (8, 5, 3)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMMA for each line
    jaw = smma(close_4h, jaw_period)
    teeth = smma(close_4h, teeth_period)
    lips = smma(close_4h, lips_period)
    
    # Apply future shifts (Alligator's "smile" - looking into future)
    jaw = np.roll(jaw, -jaw_shift)
    teeth = np.roll(teeth, -teeth_shift)
    lips = np.roll(lips, -lips_shift)
    # Set shifted values to NaN (not available yet)
    jaw[-jaw_shift:] = np.nan
    teeth[-teeth_shift:] = np.nan
    lips[-lips_shift:] = np.nan
    
    # Align to 4h timeframe (wait for bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Volume spike filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        # Simple moving average of volume
        vol_sum = np.nansum(volume[:20])  # Initialize with first 20 values
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator lines and volume MA
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + max(jaw_shift, teeth_shift, lips_shift)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for Alligator alignment (trending market)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Uptrend: Lips > Teeth > Jaw (all separated)
        is_uptrend = (lips_val > teeth_val) and (teeth_val > jaw_val)
        # Downtrend: Lips < Teeth < Jaw (all separated)
        is_downtrend = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Volume confirmation: current volume > 1.5x MA
        vol_spike = volume[i] > (1.5 * vol_ma[i])
        
        if position == 0:
            # Enter long: uptrend + volume spike
            if is_uptrend and vol_spike:
                signals[i] = size
                position = 1
            # Enter short: downtrend + volume spike
            elif is_downtrend and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens (lines intertwine) or volume drops
            if not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend weakens or volume drops
            if not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Williams_Alligator_Volume_Spike"
timeframe = "4h"
leverage = 1.0