#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ADX trend filter and volume spike.
# Uses daily ADX for trend strength, 12h Donchian channels for breakout signals,
# and volume surge for confirmation. Designed to work in both bull (breakouts above upper channel)
# and bear (breakdowns below lower channel). Target: 15-30 trades/year to avoid fee drag.
name = "12h_Donchian20_1dADX25_VolumeSpike"
timeframe = "12h"
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
    if len(df_1d) < 2:
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
    
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    for i in range(len(tr)):
        if atr[i] > 0:
            # Use Wilder's smoothing for DI
            if i < period:
                plus_di[i] = np.nan
                minus_di[i] = np.nan
            elif i == period:
                plus_di[i] = 100 * np.sum(plus_dm[1:period+1]) / (atr[i] * period)
                minus_di[i] = 100 * np.sum(minus_dm[1:period+1]) / (atr[i] * period)
            else:
                plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / (atr[i] * period))
                minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / (atr[i] * period))
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < 2 * period - 1:
            adx[i] = np.nan
        elif i == 2 * period - 1:
            adx[i] = np.nanmean(dx[period:2*period])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Calculate Donchian channels (20-period) for 12h timeframe
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i < 20:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + 1d ADX > 25 + volume spike
            if (price > highest_high[i] and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + 1d ADX > 25 + volume spike
            elif (price < lowest_low[i] and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below upper channel or ADX drops below 20
            if price < highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above lower channel or ADX drops below 20
            if price > lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals