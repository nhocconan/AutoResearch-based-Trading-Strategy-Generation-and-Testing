#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses Bill Williams' Alligator (Jaw, Teeth, Lips) to identify trend presence and direction
# Combined with 1d EMA trend filter and volume confirmation to reduce false signals
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear markets by requiring alignment between Alligator, trend, and volume
name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
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
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    close_12h = df_12h['close'].values
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw (green alignment)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Jaw > Teeth > Lips (red alignment)
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + uptrend + volume confirmation
            if bullish_alignment and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + downtrend + volume confirmation
            elif bearish_alignment and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Alligator alignment or trend reversal
            if bearish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Alligator alignment or trend reversal
            if bullish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals