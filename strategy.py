#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + 1d Trend Filter
Uses Williams Alligator (3 SMAs) to detect trend direction, enters on pullback to median line with volume spike,
filtered by 1-day EMA50 trend. Designed for low trade frequency and robustness in bull/bear markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: 3 SMAs (Jaw=13, Teeth=8, Lips=5) - all shifted forward
    # Jaw (blue) - 13-period SMMA, shifted 8 bars
    # Teeth (red) - 8-period SMMA, shifted 5 bars
    # Lips (green) - 5-period SMMA, shifted 3 bars
    # We'll use close prices for simplicity
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    lips = smma(close, 5)   # Green, 5-period
    teeth = smma(close, 8)  # Red, 8-period
    jaw = smma(close, 13)   # Blue, 13-period
    
    # Shift as per Alligator definition: Lips 3, Teeth 5, Jaw 8
    lips_shifted = np.roll(lips, 3)
    teeth_shifted = np.roll(teeth, 5)
    jaw_shifted = np.roll(jaw, 8)
    
    # Align Alligator lines to 4h
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    
    # Volume spike detection (1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Determine trend alignment: Alligator lines ordered correctly
        # In uptrend: Lips > Teeth > Jaw (green above red above blue)
        # In downtrend: Lips < Teeth < Jaw (green below red below blue)
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: pullback to Teeth (8-period) in bullish alignment + volume spike + above 1d EMA50
            if (bullish_alignment and 
                abs(price - teeth_val) < 0.005 * price and  # within 0.5% of teeth line
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: pullback to Teeth in bearish alignment + volume spike + below 1d EMA50
            elif (bearish_alignment and 
                  abs(price - teeth_val) < 0.005 * price and  # within 0.5% of teeth line
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Jaw or trend reversal
            if price < jaw_val or not bullish_alignment or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Jaw or trend reversal
            if price > jaw_val or not bearish_alignment or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_TeethPullback_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0