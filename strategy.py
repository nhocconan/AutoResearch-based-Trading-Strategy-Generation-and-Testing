#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Camarilla pivot breakout + volume confirmation + 1d ADX trend filter.
Long when price breaks above R3 with 12h volume > 1.5x 20-period average and 1d ADX > 25.
Short when price breaks below S3 with 12h volume > 1.5x 20-period average and 1d ADX > 25.
Exit when price returns to the Camarilla H3/L3 levels.
Camarilla levels from 12h provide intraday support/resistance, ADX from 1d filters for trending markets only,
volume confirmation ensures breakout validity. Designed to capture strong intraday moves in trending 6h markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Typical price = (high + low + close) / 3
        typical = (high_vals + low_vals + close_vals) / 3.0
        # Range = high - low
        rng = high_vals - low_vals
        # Camarilla levels
        H4 = typical + (rng * 1.1 / 2)
        H3 = typical + (rng * 1.1 / 4)
        L3 = typical - (rng * 1.1 / 4)
        L4 = typical - (rng * 1.1 / 2)
        return H3, L3, H4, L4
    
    H3_12h, L3_12h, H4_12h, L4_12h = camarilla_levels(high_12h, low_12h, close_12h)
    
    # Calculate 12h volume 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    def adx(high_vals, low_vals, close_vals, window):
        # True Range
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high_vals - np.roll(high_vals, 1)) > (np.roll(low_vals, 1) - low_vals),
                           np.maximum(high_vals - np.roll(high_vals, 1), 0), 0)
        dm_minus = np.where((np.roll(low_vals, 1) - low_vals) > (high_vals - np.roll(high_vals, 1)),
                            np.maximum(np.roll(low_vals, 1) - low_vals, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+
        tr_smooth = pd.Series(tr).ewm(alpha=1/window, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/window, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/window, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where((di_plus + di_minus) == 0, 0, dx)
        adx_vals = pd.Series(dx).ewm(alpha=1/window, adjust=False).mean().values
        return adx_vals
    
    adx_14_1d = adx(high_1d, low_1d, close_1d, 14)
    
    # Align all to primary timeframe (6h)
    H3_12h_aligned = align_htf_to_ltf(prices, df_12h, H3_12h)
    L3_12h_aligned = align_htf_to_ltf(prices, df_12h, L3_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_12h_aligned[i]) or 
            np.isnan(L3_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 12h 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Trend filter: 1d ADX > 25
        trending = adx_14_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above H3 with volume and trend
            if (close[i] > H3_12h_aligned[i] and 
                volume_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume and trend
            elif (close[i] < L3_12h_aligned[i] and 
                  volume_confirmed and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below L3 (opposite side)
            if close[i] < L3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above H3 (opposite side)
            if close[i] > H3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hCamarilla_H3L3_Breakout_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0