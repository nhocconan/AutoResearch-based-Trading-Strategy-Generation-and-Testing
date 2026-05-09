#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R(14) extreme reversal with 1d ADX25 trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions; ADX confirms trend strength; volume spike validates breakout.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_WilliamsR14_1dADX25_VolumeSpike"
timeframe = "4h"
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
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = (1 - alpha) * plus_dm_smooth[i-1] + alpha * plus_dm[i]
        minus_dm_smooth[i] = (1 - alpha) * minus_dm_smooth[i-1] + alpha * minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = 100 * np.where(atr > 0, plus_dm_smooth / atr, 0)
    minus_di = 100 * np.where(atr > 0, minus_dm_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.mean(dx[1:period+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Calculate Williams %R (14-period) for 4h timeframe
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    williams_r = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < 14:
            williams_r[i] = np.nan
        else:
            hh = np.max(high[i-13:i+1])
            ll = np.min(low[i-13:i+1])
            if hh - ll != 0:
                williams_r[i] = (hh - close[i]) / (hh - ll) * -100
            else:
                williams_r[i] = -50  # neutral when no range
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need 14 periods for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + 1d ADX > 25 + volume spike
            if (williams_r[i] < -80 and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + 1d ADX > 25 + volume spike
            elif (williams_r[i] > -20 and adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or ADX drops below 20
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or ADX drops below 20
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals