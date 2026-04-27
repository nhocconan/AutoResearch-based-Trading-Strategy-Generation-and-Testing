#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend confirmation and volume spike.
- Uses Williams Alligator (Jaw=13-period SMMA shifted 8, Teeth=8-period SMMA shifted 5, Lips=5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw with price above all three and volume spike
- Short when Lips < Teeth < Jaw with price below all three and volume spike
- Requires 1d EMA50 trend filter to avoid counter-trend trades
- Designed for 12h timeframe to capture multi-day moves while avoiding excessive noise
- Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator components on daily
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_1d = (high_1d := (df_1d['high'].values + df_1d['low'].values) / 2)
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Shift the lines (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_1d_shifted = np.roll(jaw_1d, 8)
    teeth_1d_shifted = np.roll(teeth_1d, 5)
    lips_1d_shifted = np.roll(lips_1d, 3)
    
    # Set NaN for shifted positions that don't have data
    jaw_1d_shifted[:8] = np.nan
    teeth_1d_shifted[:5] = np.nan
    lips_1d_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_1d_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_1d_shifted)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_1d_shifted)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average (stricter for 12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or 
            np.isnan(lips_12h[i]) or np.isnan(ema50_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price above all three AND volume spike AND above 1d EMA50
            if (lips_12h[i] > teeth_12h[i] > jaw_12h[i] and 
                close[i] > lips_12h[i] and close[i] > teeth_12h[i] and close[i] > jaw_12h[i] and
                volume_spike[i] and close[i] > ema50_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price below all three AND volume spike AND below 1d EMA50
            elif (lips_12h[i] < teeth_12h[i] < jaw_12h[i] and 
                  close[i] < lips_12h[i] and close[i] < teeth_12h[i] and close[i] < jaw_12h[i] and
                  volume_spike[i] and close[i] < ema50_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below Jaw (trend change) OR lips crosses below teeth
            if (close[i] < jaw_12h[i] or lips_12h[i] < teeth_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Jaw (trend change) OR lips crosses above teeth
            if (close[i] > jaw_12h[i] or lips_12h[i] > teeth_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_BullBear_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0