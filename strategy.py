#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla pivot R1/S1 breakout + volume confirmation + ADX trend filter.
Long when price breaks above 1d Camarilla R1 with volume confirmation and ADX > 25 (trending).
Short when price breaks below 1d Camarilla S1 with volume confirmation and ADX > 25 (trending).
Exit when price returns to the 1d Camarilla midpoint (mean reversion to pivot center).
Designed to capture institutional breakouts with volume confirmation in trending markets while avoiding false breakouts in ranging conditions.
Uses 1d timeframe for Camarilla pivot structure (reduces noise) and 4h for entry timing and trend filter.
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
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R1 = close + (range * 1.1 / 12)
    # S1 = close - (range * 1.1 / 12)
    # Mid = (R1 + S1) / 2
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have NaN due to roll, handled by min_periods later
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + (range_1d * 1.1 / 12)
    s1_1d = prev_close_1d - (range_1d * 1.1 / 12)
    mid_1d = (r1_1d + s1_1d) / 2.0
    
    # Calculate 4h ADX(14) for trend filter
    # ADX calculation requires +DI, -DI, and DX
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM, 14) / EWMA(TR, 14)
    # -DI = 100 * EWMA(-DM, 14) / EWMA(TR, 14)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX, 14)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    prev_high = high_s.shift(1)
    prev_low = low_s.shift(1)
    prev_close = close_s.shift(1)
    
    # Calculate +DM and -DM
    up_move = high_s - prev_high
    down_move = prev_low - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - prev_close)
    tr3 = abs(low_s - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate smoothed +DM, -DM, and TR using EWMA (Wilder's smoothing = alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize with first value (simple average for first period)
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx_values = adx.values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(adx_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_values[i] > 25
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and trend
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and trend
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  trending):
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

name = "4h_1dCamarilla_R1S1_Breakout_Volume_ADX25"
timeframe = "4h"
leverage = 1.0