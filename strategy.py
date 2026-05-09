#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In trending markets (1d ADX>25),
# we take long when bull power > 0 and rising, short when bear power < 0 and falling.
# Volume surge confirms institutional participation. Designed for 6h timeframe to capture
# medium-term trends while avoiding noise. Target: 12-37 trades/year to minimize fee drag.
name = "6h_ElderRay_1dADX25_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 14-period ADX for daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = (1 - alpha) * plus_dm_smooth[i-1] + alpha * plus_dm[i]
        minus_dm_smooth[i] = (1 - alpha) * minus_dm_smooth[i-1] + alpha * minus_dm[i]
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth Elder Ray components for signal generation (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Need enough data for EMA13 and ADX
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull power positive AND rising + 1d ADX > 25 + volume confirmation
            if (bull_power_smooth[i] > 0 and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and
                adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative AND falling + 1d ADX > 25 + volume confirmation
            elif (bear_power_smooth[i] < 0 and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and
                  adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative OR ADX drops below 20
            if bull_power_smooth[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive OR ADX drops below 20
            if bear_power_smooth[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals