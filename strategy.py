#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h ADX25 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with 12h ADX > 25 and volume > 1.5x average.
# Short when price breaks below Donchian lower band with 12h ADX > 25 and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Donchian channels provide structural breakout levels. 12h ADX ensures we trade only when intermediate trend is strong.
# Volume spike confirms institutional participation. Works in bull markets via upward breaks and in bear markets via downward breaks.

name = "4h_Donchian20_12hADX25_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for ADX25 trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    if len(close_12h) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 12h data
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing, equivalent to EMA with alpha=1/14)
    period = 14
    alpha = 1.0 / period
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Initialize first values
    if not np.isnan(tr[period]):
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
    
    # Wilder's smoothing
    for i in range(period+1, len(tr)):
        if not np.isnan(tr[i]):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i-1]
            minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i-1]
    
    # Calculate +DI and -DI
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(period, len(tr)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(dx, np.nan)
    for i in range(2*period-1, len(dx)):
        if not np.isnan(dx[i]):
            if np.isnan(adx[i-1]):
                adx[i] = np.nanmean(dx[period-1:i+1])
            else:
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align 12h ADX to 4h timeframe (wait for 12h bar to close)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback + 20, 30), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band with 12h ADX > 25 and volume spike
            if (close[i] > upper_band[i] and 
                adx_12h_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with 12h ADX > 25 and volume spike
            elif (close[i] < lower_band[i] and 
                  adx_12h_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower band (reversal signal)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper band (reversal signal)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals