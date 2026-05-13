#!/usr/bin/env python3
"""
1d_Williams_Alligator_Volume_Trend
Hypothesis: Use Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and filter for trend alignment. Enter long when price > Teeth > Jaw and Teeth slope > 0, short when price < Teeth < Jaw and Teeth slope < 0, with volume confirmation (volume > 1.5x 20-day average). Exit when price crosses Teeth or slope changes. Designed for 1d timeframe to limit trades (<20/year) and avoid fee drag. Works in both bull (captures trends) and bear (avoids false signals via slope filter).
"""

name = "1d_Williams_Alligator_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = np.roll(jaw_raw.values, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # fill shifted values with NaN
    
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = np.roll(teeth_raw.values, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = np.roll(lips_raw.values, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Teeth slope for trend strength (3-period change)
    teeth_slope = np.diff(teeth_aligned, prepend=np.nan)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(teeth_slope[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price > Teeth > Jaw AND Teeth slope > 0 (bullish alignment) + volume spike
            if close[i] > teeth_aligned[i] > jaw_aligned[i] and teeth_slope[i] > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Teeth < Jaw AND Teeth slope < 0 (bearish alignment) + volume spike
            elif close[i] < teeth_aligned[i] < jaw_aligned[i] and teeth_slope[i] < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth OR Teeth slope turns negative
            if close[i] < teeth_aligned[i] or teeth_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth OR Teeth slope turns positive
            if close[i] > teeth_aligned[i] or teeth_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals