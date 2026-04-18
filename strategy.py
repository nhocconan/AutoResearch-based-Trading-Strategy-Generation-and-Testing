#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend_With_Volume_Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on daily timeframe defines trend direction.
Trades occur on 12h timeframe when price crosses Lips with volume confirmation.
Uses Williams Alligator crossover signals filtered by volume > 1.5x 20-period average.
Designed for 12h timeframe with ~15-30 trades/year to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines (Smoothed Medians)"""
    median_price = (high + low) / 2
    
    # Jaw (Blue) - 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=jaw_period, center=False).mean().shift(8)
    # Teeth (Red) - 8-period SMMA, shifted 5 bars forward  
    teeth = pd.Series(median_price).rolling(window=teeth_period, center=False).mean().shift(5)
    # Lips (Green) - 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=lips_period, center=False).mean().shift(3)
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Williams Alligator for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    jaw_1d, teeth_1d, lips_1d = williams_alligator(high_1d, low_1d, close_1d)
    
    # Align Daily Alligator lines to 12h timeframe (wait for daily bar close)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 12h volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for volume MA and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        # Alligator alignment: Jaw > Teeth > Lips = Uptrend, Jaw < Teeth < Lips = Downtrend
        uptrend = jaw > teeth and teeth > lips
        downtrend = jaw < teeth and teeth < lips
        
        if position == 0:
            # Long: price crosses above Lips in uptrend with volume
            if price > lips and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Lips in downtrend with volume
            elif price < lips and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below Teeth or trend changes
            if price < teeth or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above Teeth or trend changes
            if price > teeth or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0