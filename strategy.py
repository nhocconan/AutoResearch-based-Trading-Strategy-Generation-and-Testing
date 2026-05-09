#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d ADX25 trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions (above -20 or below -80).
# Trend filter ensures we trade in direction of daily trend.
# Volume spike confirms momentum.
# Designed for 12h timeframe to target 12-37 trades/year.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
name = "12h_WilliamsR14_1dADX25_VolumeSpike"
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
    
    # Calculate +DI and -DI
    plus_di = np.zeros_like(atr)
    minus_di = np.zeros_like(atr)
    for i in range(len(atr)):
        if atr[i] > 0:
            plus_di[i] = 100 * np.sum(plus_dm[max(0, i-period+1):i+1]) / (atr[i] * period)
            minus_di[i] = 100 * np.sum(minus_dm[max(0, i-period+1):i+1]) / (atr[i] * period)
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
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
        if i < 14:
            williams_r[i] = np.nan
        else:
            highest_high_14 = np.max(high[i-13:i+1])
            lowest_low_14 = np.min(low[i-13:i+1])
            if highest_high_14 != lowest_low_14:
                williams_r[i] = -100 * (highest_high_14 - close[i]) / (highest_high_14 - lowest_low_14)
            else:
                williams_r[i] = -50  # Avoid division by zero
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need 14 periods for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) + 1d ADX > 25 + volume spike
            if (williams_r[i] < -80 and adx[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) + 1d ADX > 25 + volume spike
            elif (williams_r[i] > -20 and adx[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (return to midpoint) or ADX drops below 20
            if williams_r[i] > -50 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (return to midpoint) or ADX drops below 20
            if williams_r[i] < -50 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals