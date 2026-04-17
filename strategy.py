#!/usr/bin/env python3
"""
12h Williams Alligator + 1d Volume Spike + ADX Trend Filter
Long: Jaw < Teeth < Lips (bullish alignment) + volume > 2x 12h volume SMA(20) + ADX(1d) > 25
Short: Jaw > Teeth > Lips (bearish alignment) + volume > 2x 12h volume SMA(20) + ADX(1d) > 25
Exit: Opposite Alligator alignment or ADX < 20
Williams Alligator uses SMAs of median price with specific offsets to identify trends.
Designed to catch strong trends with volume confirmation and avoid choppy markets.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Jaw (blue line) - 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = np.roll(jaw_raw.values, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    # Teeth (red line) - 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = np.roll(teeth_raw.values, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips (green line) - 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = np.roll(lips_raw.values, lips_shift)
    lips[:lips_shift] = np.nan
    
    # Get 1d data for volume SMA and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 12h volume SMA(20)
    vol_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_sma_12h_aligned = vol_sma_12h  # already 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 20, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_sma_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_12h_aligned[i]
        adx_val = adx_aligned[i]
        
        # Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish alignment + volume spike + strong trend
            if bullish_alignment and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + strong trend
            elif bearish_alignment and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or weak trend
            if bearish_alignment or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or weak trend
            if bullish_alignment or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0