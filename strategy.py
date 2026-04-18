#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and 1d ADX Trend Filter
Reversals at Camarilla R3/S3 levels during high volume, filtered by daily ADX trend strength
Designed to capture mean-reversion in ranging markets and trend continuation in strong trends
Works in both bull and bear markets by adapting to volatility regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), R2 = C + ((H-L) * 1.1/2), R1 = C + ((H-L) * 1.05/2)
    # S1 = C - ((H-L) * 1.05/2), S2 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.25/2), S4 = C - ((H-L) * 1.5/2)
    cam_multiplier = (high_1d - low_1d) * 0.005  # (H-L) * 0.5% for inner levels
    r3 = close_1d + (high_1d - low_1d) * 1.25 / 2  # R3 level
    s3 = close_1d - (high_1d - low_1d) * 1.25 / 2  # S3 level
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1d data for ADX trend filter
    # Calculate ADX components: +DI, -DI, DX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA-like)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for ADX calculation
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        adx_value = adx_aligned[i]
        
        if position == 0:
            # Long reversal at S3 with volume spike (works in ranging and weak trends)
            if (price <= s3_level and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short reversal at R3 with volume spike
            elif (price >= r3_level and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R3 (mean reversion target) or ADX weakens
            if price >= r3_level or adx_value < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S3 or ADX weakens
            if price <= s3_level or adx_value < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Reversal_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0