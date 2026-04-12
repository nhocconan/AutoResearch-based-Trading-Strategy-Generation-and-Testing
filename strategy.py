#!/usr/bin/env python3
"""
6h_12h_1d_williams_alligator_trend_v1
Hypothesis: Williams Alligator (3 SMAs) on 1d for trend, 12h for momentum confirmation via price > SMA3, 
and 6h for entry on pullback to SMA2 with volume confirmation. Works in bull/bear by trading with 1d trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_12h_1d_williams_alligator_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period), Lips (5-period)
    # SMMA: smoothed moving average (like Wilder's smoothing)
    def smoothed_ma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smoothed_ma(close_1d, 13)   # Blue line
    teeth = smoothed_ma(close_1d, 8)   # Red line
    lips = smoothed_ma(close_1d, 5)    # Green line
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 12h data for momentum confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h SMA3 for momentum (price above SMA3 = bullish momentum)
    sma3_12h = pd.Series(close_12h).rolling(window=3, min_periods=3).mean().values
    sma3_12h_aligned = align_htf_to_ltf(prices, df_12h, sma3_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(sma3_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: Alligator aligned (Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Momentum: price above/below 12h SMA3
        price_above_sma3 = close[i] > sma3_12h_aligned[i]
        price_below_sma3 = close[i] < sma3_12h_aligned[i]
        
        # Entry conditions
        if bullish_aligned and price_above_sma3 and vol_confirm[i] and position != 1:
            # Long: enter on pullback to lips (green line) in uptrend
            if close[i] <= lips_aligned[i] * 1.005:  # Within 0.5% of lips
                position = 1
                signals[i] = 0.25
        elif bearish_aligned and price_below_sma3 and vol_confirm[i] and position != -1:
            # Short: enter on pullback to lips in downtrend
            if close[i] >= lips_aligned[i] * 0.995:  # Within 0.5% of lips
                position = -1
                signals[i] = -0.25
        # Exit: trend change or opposite lip touch
        elif position == 1 and (not bullish_aligned or close[i] >= teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_aligned or close[i] <= teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals