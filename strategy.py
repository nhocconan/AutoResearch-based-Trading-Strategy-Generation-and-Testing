#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + ADX Trend Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction.
# Enters long when Lips > Teeth > Jaw and price > Lips, short when opposite.
# Requires volume > 1.5x median volume (20-period) and ADX > 25 for trend confirmation.
# Exits when Alligator lines cross in opposite direction or ADX < 20 (ranging market).
# Designed for 12h timeframe to avoid overtrading and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator (13,8,5 SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator lines
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Shift to avoid look-ahead (Alligator uses future data if not shifted)
    jaw = np.roll(jaw, 8)   # Jaw shifted by 8
    teeth = np.roll(teeth, 5) # Teeth shifted by 5
    lips = np.roll(lips, 3)  # Lips shifted by 3
    
    # Load 1d data for ADX trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
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
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Alligator and ADX to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > Lips
        #           AND volume confirmation AND ADX > 25 (trending market)
        if (lips_aligned[i] > teeth_aligned[i] and 
            teeth_aligned[i] > jaw_aligned[i] and
            close[i] > lips_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < Lips
        #            AND volume confirmation AND ADX > 25 (trending market)
        elif (lips_aligned[i] < teeth_aligned[i] and 
              teeth_aligned[i] < jaw_aligned[i] and
              close[i] < lips_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator lines cross in opposite direction OR ADX < 20 (ranging market)
        elif position == 1 and (lips_aligned[i] < teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0