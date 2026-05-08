#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with volume confirmation and ADX trend filter.
# Williams %R identifies overbought/oversold conditions; enters on reversals from extremes.
# Long when %R crosses above -80 from below with volume confirmation and ADX > 25 (trending).
# Short when %R crosses below -20 from above with volume confirmation and ADX > 25.
# Exit when %R crosses -50 (mean reversion) or ADX < 20 (range market).
# Designed for low trade frequency (20-40/year) to avoid fee drift. Works in trending markets.

name = "4h_1dWilliamsR_ADX_Volume"
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
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    period = 14
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i < period - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_1d[i - period + 1:i + 1])
            lowest_low[i] = np.min(low_1d[i - period + 1:i + 1])
    
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
            denominator = highest_high[i] - lowest_low[i]
            if denominator != 0:
                williams_r[i] = (highest_high[i] - close_1d[i]) / denominator * -100
            else:
                williams_r[i] = -50  # Avoid division by zero
    
    # Calculate 1-day ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (14-period)
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initial values (first 14 periods)
    if len(tr) >= period + 1:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full_like(dm_plus_smooth, np.nan)
    minus_di = np.full_like(dm_minus_smooth, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(len(tr)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = dm_plus_smooth[i] / atr[i] * 100
            minus_di[i] = dm_minus_smooth[i] / atr[i] * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 2 * period:
        # Initial ADX (average of first 14 DX values after period)
        adx[2*period - 1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below, ADX > 25, volume confirmation
            if (i > start_idx and 
                williams_r_aligned[i-1] < -80 and williams_r_aligned[i] >= -80 and
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above, ADX > 25, volume confirmation
            elif (i > start_idx and 
                  williams_r_aligned[i-1] > -20 and williams_r_aligned[i] <= -20 and
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or ADX < 20 (range market)
            if (williams_r_aligned[i] > -50) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or ADX < 20 (range market)
            if (williams_r_aligned[i] < -50) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals