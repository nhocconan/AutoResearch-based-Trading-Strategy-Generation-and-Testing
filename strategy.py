#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla pivot from daily + volume confirmation
# Hypothesis: Camarilla levels (R3/S3, R4/S4) act as strong support/resistance. 
# Fade at R3/S3 with rejection, breakout continuation at R4/S4. Volume confirms institutional interest.
# Works in both bull/bear: mean reversion in ranges, trend following in breakouts.
# Target: 15-35 trades/year to minimize fee drag.
name = "6h_camarilla_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # S4 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero in range
    prev_range = prev_high - prev_low
    prev_range = np.where(prev_range == 0, 1e-10, prev_range)
    
    # Calculate levels
    r4 = prev_close + prev_range * 1.1 / 2
    r3 = prev_close + prev_range * 1.1 / 4
    s3 = prev_close - prev_range * 1.1 / 4
    s4 = prev_close - prev_range * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 day for lookback)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period volume moving average for confirmation
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (support broken) OR above R4 (take profit at extreme)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above R3 (resistance broken) OR below S4 (take profit at extreme)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at R3/S3: short at R3 with rejection, long at S3 with rejection
            # Breakout continuation: long above R4, short below S4
            
            # Enter long: price closes above S3 with volume (fade) OR above R4 with volume (breakout)
            if ((close[i] > s3_aligned[i] and close[i] <= r3_aligned[i] and vol_confirm) or  # Fade at S3
                (close[i] > r4_aligned[i] and vol_confirm)):  # Breakout above R4
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below R3 with volume (fade) OR below S4 with volume (breakout)
            elif ((close[i] < r3_aligned[i] and close[i] >= s3_aligned[i] and vol_confirm) or  # Fade at R3
                  (close[i] < s4_aligned[i] and vol_confirm)):  # Breakdown below S4
                position = -1
                signals[i] = -0.25
    
    return signals