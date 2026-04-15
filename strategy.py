#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + ATR Filter
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction.
# Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish alignment).
# Volume confirmation requires > 1.5x 20-bar median volume.
# ATR-based exit: close position when price moves against position by 1.5x ATR.
# Designed to work in trending markets (both bull and bear) with clear directional signals.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Williams Alligator
    def smma(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    df_1d = get_htf_data(prices, '1d')
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = smma(median_price_1d, 13)  # Blue line
    teeth = smma(median_price_1d, 8)  # Red line
    lips = smma(median_price_1d, 5)   # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ATR for exit condition
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = high[0] - low[0]
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_vals = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr_vals
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_vals = atr(high, low, close, 14)
    
    # Volume confirmation
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(atr_vals[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Check for Alligator alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Long: Bullish alignment + volume spike
        if bullish_alignment and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: Bearish alignment + volume spike
        elif bearish_alignment and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: Price moves against position by 1.5x ATR
        elif signals[i-1] == 0.25 and close[i] < close[i-1] - 1.5 * atr_vals[i]:
            signals[i] = 0.0
        elif signals[i-1] == -0.25 and close[i] > close[i-1] + 1.5 * atr_vals[i]:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_Volume_ATR"
timeframe = "12h"
leverage = 1.0