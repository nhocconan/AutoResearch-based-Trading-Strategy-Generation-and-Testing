#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Williams %R with 14-period and 1-day ADX for trend filter.
# Long when weekly Williams %R < -80 (oversold) and daily ADX > 25 (trending).
# Short when weekly Williams %R > -20 (overbought) and daily ADX > 25 (trending).
# Exit when weekly Williams %R crosses back to neutral (-50).
# Designed for low trade frequency (10-20/year) to avoid fee dust. Works in trending markets via ADX filter.

name = "1d_1wWilliamsR_ADX_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    highest_high = np.zeros_like(close_1w)
    lowest_low = np.zeros_like(close_1w)
    
    for i in range(len(close_1w)):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_1w[i-13:i+1])
            lowest_low[i] = np.min(low_1w[i-13:i+1])
    
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close_1w) / (highest_high - lowest_low),
        -50
    )
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (14-period)
    def wilder_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(13, len(dx)):  # First 13 values are NaN from Wilder smoothing
        if i == 13:
            adx[i] = np.nanmean(dx[:14])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1w Williams %R to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) and ADX > 25 (trending)
            if (williams_r_aligned[i] < -80 and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) and ADX > 25 (trending)
            elif (williams_r_aligned[i] > -20 and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals