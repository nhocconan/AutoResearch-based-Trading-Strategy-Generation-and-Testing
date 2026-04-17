#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla R3/S3 breakout + volume confirmation + 4h ADX25 trend filter.
Long when price breaks above 1d Camarilla R3 with volume confirmation and ADX > 25 (strong uptrend).
Short when price breaks below 1d Camarilla S3 with volume confirmation and ADX > 25 (strong downtrend).
Exit when price returns to the 1d Camarilla midpoint (mean reversion to pivot center).
Designed to capture strong institutional breakouts with volume confirmation while avoiding chop.
Uses 1d for structure (Camarilla pivots) and 4h for entry timing and trend filter.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3, S3, midpoint)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Simplified: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Midpoint = (R3 + S3)/2 = close
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d * 1.1 / 4.0
    s3_1d = close_1d - 1.1 * range_1d * 1.1 / 4.0
    mid_1d = close_1d  # Camarilla midpoint is the previous day's close
    
    # Calculate 4h ADX(14) for trend filter
    # ADX = DX smoothed, DX = |DI+ - DI-| / (DI+ + DI-) * 100
    # DI+ = (Today's High - Yesterday's High) if > 0 and > (Yesterday's Low - Today's Low), else 0
    # DI- = (Yesterday's Low - Today's Low) if > 0 and > (Today's High - Yesterday's High), else 0
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(adx_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        # Trend filter: ADX > 25 (strong trend)
        strong_trend = adx_values[i] > 25
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume and strong uptrend
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirmed and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume and strong downtrend
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirmed and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1d Camarilla midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1d Camarilla midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R3S3_Breakout_Volume_ADX25"
timeframe = "4h"
leverage = 1.0