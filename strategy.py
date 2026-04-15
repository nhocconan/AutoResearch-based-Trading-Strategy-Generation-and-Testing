#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + ADX Trend Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with volume confirmation and ADX > 25.
# Enters short when Lips < Teeth < Jaw (bearish alignment) with volume confirmation and ADX > 25.
# Exits when Alligator alignment breaks or ADX < 20 (ranging market).
# Designed for 12h timeframe to capture medium-term trends in BTC/ETH with low trade frequency.
# Williams Alligator is effective in trending markets and avoids whipsaws in ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for Williams Alligator (13,8,5 SMAs of median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
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
    
    # Align Williams Alligator and ADX to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Calculate volume confirmation (current volume > 1.5x median of past 20 periods)
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_ok = volume[i] > 1.5 * vol_median
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Long entry: Bullish alignment + volume confirmation + ADX > 25
        if (bullish_alignment and volume_ok and adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + volume confirmation + ADX > 25
        elif (bearish_alignment and volume_ok and adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator alignment breaks or ADX < 20 (ranging market)
        elif position == 1 and (not bullish_alignment or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0