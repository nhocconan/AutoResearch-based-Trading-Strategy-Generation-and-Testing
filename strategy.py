#!/usr/bin/env python3
"""
Hypothesis: 1-day Williams Alligator with 1-week trend filter.
Long when jaw > teeth > lips (bullish alignment) and price > close[0] (first bar of week).
Short when jaw < teeth < lips (bearish alignment) and price < close[0].
Exit when alignment breaks or price crosses weekly open.
Williams Alligator identifies trend phases; weekly open provides institutional reference.
Works in bull markets by catching trends, in bear markets by avoiding false signals via weekly filter.
"""

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
    
    # Load 1-day data for Alligator - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 1-day timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly open (first bar of each week)
    weekly_open = df_1w['open'].values
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, weekly_open)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: jaw > teeth > lips
            bullish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            # Bearish alignment: jaw < teeth < lips
            bearish = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
            
            if bullish and close[i] > weekly_open_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif bearish and close[i] < weekly_open_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: alignment breaks or price below weekly open
                bullish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
                if not bullish or close[i] < weekly_open_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: alignment breaks or price above weekly open
                bearish = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
                if not bearish or close[i] > weekly_open_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsAlligator_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0