#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Volume Spike + 1d Trend Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) on 4h to identify trend direction.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with volume spike.
# Enters short when Lips < Teeth < Jaw (bearish alignment) with volume spike.
# Uses 1d ADX > 25 to filter for trending markets only.
# Williams Alligator is effective in catching trends while avoiding whipsaws in ranging markets.
# Volume spike confirms institutional participation.
# Designed for 4-6 trades per month (~50-75/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Williams Alligator (13,8,5 SMAs of median price)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate median price for 4h
    median_price_4h = (high_4h + low_4h) / 2.0
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMAs of median price
    jaw = pd.Series(median_price_4h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_4h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_4h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 4h timeframe (no additional delay needed as they are based on current bar)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
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
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Volume spike: current volume > 2x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long entry: bullish alignment + volume spike + ADX > 25
        if bullish_alignment and volume_spike and (adx_aligned[i] > 25) and (position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + volume spike + ADX > 25
        elif bearish_alignment and volume_spike and (adx_aligned[i] > 25) and (position >= 0):
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

name = "4h_WilliamsAlligator_Volume_ADX"
timeframe = "4h"
leverage = 1.0