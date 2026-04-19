#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA13 trend filter and volume confirmation
# Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends and avoid chop
# 1d EMA13 provides higher timeframe bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms strength
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_Alligator_1dEMA13_Volume"
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
    
    # 1d EMA13 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Williams Alligator calculation on 4h
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Smoothed Moving Average (SMMA) - similar to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA for median price (typical price)
    median_price = (high + low) / 2
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (shift right/pad with NaN for simplicity in alignment)
    # We'll handle the shift by using the values directly and adjusting logic
    # For simplicity in this implementation, we'll use the values as-is and
    # rely on the convergence/divergence logic
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + max(jaw_shift, teeth_shift, lips_shift)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals:
        # Mouth open (trending): Lips > Teeth > Jaw (bullish) OR Lips < Teeth < Jaw (bearish)
        # Mouth closed (chopping): intertwined lines
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish alignment + above 1d EMA13 + volume confirmation
            if (bullish_alignment and 
                close[i] > ema_13_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + below 1d EMA13 + volume confirmation
            elif (bearish_alignment and 
                  close[i] < ema_13_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if alignment breaks down or price breaks below 1d EMA13
            if not bullish_alignment or close[i] < ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if alignment breaks up or price breaks above 1d EMA13
            if not bearish_alignment or close[i] > ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals