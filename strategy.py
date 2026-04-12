#!/usr/bin/env python3
"""
4h_1d_Alligator_Trend_Filter_V1
Hypothesis: Uses Williams Alligator (three SMAs) on 1d timeframe as a trend filter on 4h chart.
Enter long when price > Alligator Jaw (13-period SMA) and Alligator is bullish (Teeth > Lips).
Enter short when price < Alligator Jaw and Alligator is bearish (Teeth < Lips).
Requires 4h volume above 20-period average to confirm momentum.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in bull via buying uptrend pullbacks, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Alligator_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ALLIGATOR ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    close_1d_series = pd.Series(close_1d)
    jaw = close_1d_series.rolling(window=13, min_periods=13).mean().values
    teeth = close_1d_series.rolling(window=8, min_periods=8).mean().values
    lips = close_1d_series.rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 4H VOLUME FILTER ===
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator trend conditions
        alligator_bullish = teeth_aligned[i] > lips_aligned[i]
        alligator_bearish = teeth_aligned[i] < lips_aligned[i]
        price_above_jaw = close[i] > jaw_aligned[i]
        price_below_jaw = close[i] < jaw_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > vol_ma[i]
        
        # Entry conditions
        long_setup = alligator_bullish and price_above_jaw and volume_ok
        short_setup = alligator_bearish and price_below_jaw and volume_ok
        
        # Exit when trend changes or volume fails
        exit_long = not (alligator_bullish and price_above_jaw and volume_ok)
        exit_short = not (alligator_bearish and price_below_jaw and volume_ok)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals