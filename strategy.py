#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation
# - Williams Alligator: Jaw (13 SMMA shifted 8), Teeth (8 SMMA shifted 5), Lips (5 SMMA shifted 3)
# - Trend: Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend
# - Volume: 1d volume > 1.5x 20-period average for confirmation
# - Entry: Alligator aligned + volume confirmation
# - Exit: Opposite Alligator alignment
# - Session filter: 08:00-20:00 UTC
# - Position size: 0.25
# - Designed for 12h timeframe to target 12-37 trades/year
# - Williams Alligator captures trends with built-in smoothing to reduce whipsaw

name = "12h_WilliamsAlligator_1dVolume_v1"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams Alligator components on close prices
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Ensure enough data for all indicators (max shift + period)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Alligator alignment
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Look for long entry: bullish alignment + volume
            if bullish and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish alignment + volume
            elif bearish and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish alignment
            if bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish alignment
            if bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals