#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaws/Teeth/Lips) with 1d ADX trend filter and volume confirmation.
# Long when Lips > Teeth > Jaws (bullish alignment) AND price closes above Lips AND 1d ADX > 25 AND 12h volume > 1.3x 34-period average.
# Short when Lips < Teeth < Jaws (bearish alignment) AND price closes below Lips AND 1d ADX > 25 AND 12h volume > 1.3x 34-period average.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Jaws) or price crosses Lips in opposite direction.
# Uses Alligator for trend detection with ADX filter to avoid ranging markets, targeting 50-120 trades over 4 years.

name = "12h_Alligator_1dADX_Volume"
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
    
    # 12h Williams Alligator (Smoothed Moving Average - SMMA)
    # Jaws: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        # Smoothed Moving Average: first value is SMA, then SMMA = (prev*(period-1) + current)/period
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # Initial SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent SMMA values
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift jaws and teeth as per Alligator definition
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    # First 8 values of jaws and first 5 of teeth are invalid after roll
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    
    # 12h volume filter: current volume > 1.3x 34-period average
    vol_ma34 = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    volume_filter = volume > (1.3 * vol_ma34)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d data
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
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Alligator and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaws[i])
        
        if position == 0:
            # Long conditions: bullish alignment, price above lips, ADX > 25, volume spike
            long_cond = bullish_alignment and (close[i] > lips[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: bearish alignment, price below lips, ADX > 25, volume spike
            short_cond = bearish_alignment and (close[i] < lips[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: alignment breaks or price crosses below lips
            if not bullish_alignment or (close[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: alignment breaks or price crosses above lips
            if not bearish_alignment or (close[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals