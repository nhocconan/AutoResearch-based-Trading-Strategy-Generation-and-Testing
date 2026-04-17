#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams %R (14-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(high_12h, np.nan)
    lowest_low = np.full_like(low_12h, np.nan)
    period = 14
    
    for i in range(len(high_12h)):
        if i >= period - 1:
            highest_high[i] = np.max(high_12h[i - period + 1:i + 1])
            lowest_low[i] = np.min(low_12h[i - period + 1:i + 1])
        else:
            highest_high[i] = np.max(high_12h[0:i + 1])
            lowest_low[i] = np.min(low_12h[0:i + 1])
    
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close_12h) / denominator) * -100, -50)
    
    # === 1d ADX (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(plus_dm, np.nan)
    minus_di = np.full_like(minus_dm, np.nan)
    
    # Initialize first values
    atr[period - 1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    for i in range(period, len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    # Smooth DX to get ADX
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # === Align indicators to 6h timeframe ===
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Williams %R oversold (< -80) + ADX > 25 (trending) + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R overbought (> -20) + ADX > 25 (trending) + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum fading) OR ADX < 20 (trend weak)
            if (williams_r_aligned[i] > -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum fading) OR ADX < 20 (trend weak)
            if (williams_r_aligned[i] < -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADX_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0