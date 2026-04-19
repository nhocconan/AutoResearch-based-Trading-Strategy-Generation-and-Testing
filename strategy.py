#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX trend filter and volume confirmation
# Alligator uses three SMAs (Jaw, Teeth, Lips) to detect trends and convergence/divergence
# In trending markets, the lines are ordered and separated; in ranging, they intertwine
# ADX > 25 filters for trending conditions to avoid whipsaws in sideways markets
# Volume confirmation ensures breakouts have conviction
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries
name = "4h_WilliamsAlligator_1dADX_Volume"
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
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smoothed_avg(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values are smoothed
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr = smoothed_avg(tr, period)
    dm_plus_smooth = smoothed_avg(dm_plus, period)
    dm_minus_smooth = smoothed_avg(dm_minus, period)
    
    # Avoid division by zero
    dm_plus_safe = np.where(atr == 0, 1e-10, dm_plus_smooth)
    dm_minus_safe = np.where(atr == 0, 1e-10, dm_minus_smooth)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    di_plus = 100 * dm_plus_safe / atr_safe
    di_minus = 100 * dm_minus_safe / atr_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = smoothed_avg(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 4h
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Shift the lines as per Williams Alligator specification
    jaw = np.roll(jaw, int(jaw_period/2))
    teeth = np.roll(teeth, int(teeth_period/2))
    lips = np.roll(lips, int(lips_period/2))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + 5  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume confirmation
            if (lips[i] > teeth[i] > jaw[i] and 
                adx_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume confirmation
            elif (lips[i] < teeth[i] < jaw[i] and 
                  adx_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator lines cross (Lips < Teeth) or ADX drops below 20
            if (lips[i] < teeth[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator lines cross (Lips > Teeth) or ADX drops below 20
            if (lips[i] > teeth[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals