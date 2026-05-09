#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX50 trend filter and volume spike.
# Uses weekly ADX for strong trend detection, daily Donchian channels for breakout signals,
# and volume surge for confirmation. Designed to work in both bull (breakouts above upper channel)
# and bear (breakdowns below lower channel) with low trade frequency to avoid fee drag.
name = "1d_Donchian20_1wADX50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 50-period ADX for weekly timeframe
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 50
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
    
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Calculate Donchian channels (20-period) for 1d timeframe
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i < 20:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
        else:
            upper_channel[i] = np.max(high[i-19:i+1])
            lower_channel[i] = np.min(low[i-19:i+1])
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + 1w ADX > 50 + volume spike
            if (price > upper_channel[i] and adx_aligned[i] > 50 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + 1w ADX > 50 + volume spike
            elif (price < lower_channel[i] and adx_aligned[i] > 50 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below upper channel or ADX drops below 30
            if price < upper_channel[i] or adx_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above lower channel or ADX drops below 30
            if price > lower_channel[i] or adx_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals