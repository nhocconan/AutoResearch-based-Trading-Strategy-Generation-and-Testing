#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with 1-day ADX(14) trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions; ADX ensures trades only in trending markets.
# Volume spike confirms breakout strength. Designed for low trade frequency (<15/year) to avoid fee drag.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
name = "12h_WilliamsR14_1dADX14_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
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
    
    plus_di = 100 * np.where(atr > 0, 
                             np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(plus_dm)] / atr, 0)
    minus_di = 100 * np.where(atr > 0,
                              np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(minus_dm)] / atr, 0)
    
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
    
    # Calculate Williams %R (14-period) for 12h timeframe
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    williams_r = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < 13:  # Need 14 periods (0-13 inclusive)
            williams_r[i] = np.nan
        else:
            hh = np.max(high[i-13:i+1])
            ll = np.min(low[i-13:i+1])
            if hh - ll != 0:
                williams_r[i] = (hh - close[i]) / (hh - ll) * -100
            else:
                williams_r[i] = -50  # Avoid division by zero
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.8x 20-period EMA (balanced threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) + ADX > 14 + volume spike
            if (williams_r[i] < -80 and adx_aligned[i] > 14 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) + ADX > 14 + volume spike
            elif (williams_r[i] > -20 and adx_aligned[i] > 14 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) or ADX drops below 10
            if williams_r[i] > -20 or adx_aligned[i] < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) or ADX drops below 10
            if williams_r[i] < -80 or adx_aligned[i] < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals