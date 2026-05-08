#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend strength and direction.
# In trending markets (price outside Alligator mouth), we trade in direction of 1d EMA50 trend.
# Volume spike confirms breakout strength. Designed for low-frequency trades (<150 total) to minimize fee drift.
# Works in both bull and bear markets by aligning with higher timeframe trend.

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = median_price.rolling(window=8, min_periods=8).mean().values   # Red line
    lips = median_price.rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 trend filter
    ema50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2.0x 20-period EMA on 12h)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and Alligator have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above Alligator (teeth > jaw) with 1d uptrend and volume spike
            if (close[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below Alligator (teeth < jaw) with 1d downtrend and volume spike
            elif (close[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Alligator mouth (teeth < lips) or trend fails
            if (teeth_aligned[i] < lips_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Alligator mouth (teeth > lips) or trend fails
            if (teeth_aligned[i] > lips_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals