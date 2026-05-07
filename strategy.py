#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 12h EMA50 trend filter and volume spike confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA50 AND volume > 2x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA50 AND volume > 2x 20-period average.
# Exit when alignment breaks or price crosses 12h EMA50 in opposite direction.
# Designed for 6h timeframe with low trade frequency (target: 10-25/year) to avoid fee drag.
# Uses 12h EMA50 for trend filter to avoid counter-trend trades. Williams Alligator provides trend confirmation.
# Volume spike ensures participation and avoids low-conviction moves.
name = "6h_WilliamsAlligator_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift)
    def smma(arr, period):
        # Smoothed Moving Average: first value is SMA, then smoothed
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift the lines as per Williams Alligator
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_align = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_align = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
            
            # Long conditions: bullish alignment, price > 12h EMA50, volume spike
            long_cond = bullish_align and (close[i] > ema50_12h_aligned[i]) and volume_spike[i]
            # Short conditions: bearish alignment, price < 12h EMA50, volume spike
            short_cond = bearish_align and (close[i] < ema50_12h_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price < 12h EMA50
            bearish_align = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
            if bearish_align or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price > 12h EMA50
            bullish_align = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
            if bullish_align or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals