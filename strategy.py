#!/usr/bin/env python3
# 6h_1d_adx_volume_momentum_v1
# Hypothesis: 6-hour ADX > 25 + volume > 1.5x 20-period average + price > 50-period SMA for long,
# opposite for short. Uses 1-day trend filter: only take long if price > 1-day 200-period SMA,
# only short if price < 1-day 200-period SMA. Designed to capture strong trends with volume
# confirmation while avoiding counter-trend trades. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_volume_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX components
    period_adx = 14
    # True Range
    tr0 = high[1:] - low[1:]
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr0, np.maximum(tr1, tr2))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0 (nan)
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, period_adx)
    plus_dm_smooth = smooth_wilder(plus_dm, period_adx)
    minus_dm_smooth = smooth_wilder(minus_dm, period_adx)
    
    # Directional Indicators
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period_adx:
        adx[2*period_adx-2] = np.nanmean(dx[period_adx:2*period_adx-1])
        for i in range(2*period_adx-1, len(dx)):
            adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    # Volume average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    for i in range(vol_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_period+1:i+1])
    
    # Price SMA
    sma_period = 50
    sma = np.full_like(close, np.nan)
    for i in range(sma_period-1, n):
        sma[i] = np.mean(close[i-sma_period+1:i+1])
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day 200-period SMA
    sma_200_1d = np.full_like(close_1d, np.nan)
    for i in range(199, len(close_1d)):
        sma_200_1d[i] = np.mean(close_1d[i-199:i+1])
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(sma[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(sma_200_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: ADX < 20 or volume < average or price < SMA or trend fails
            if adx[i] < 20 or volume[i] < vol_ma[i] or close[i] < sma[i] or close[i] < sma_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: ADX < 20 or volume < average or price > SMA or trend fails
            if adx[i] < 20 or volume[i] < vol_ma[i] or close[i] > sma[i] or close[i] > sma_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Strong trend with volume confirmation
            # Long: ADX > 25, volume > 1.5x average, price > SMA, and price > 1-day 200 SMA
            if (adx[i] > 25 and volume[i] > 1.5 * vol_ma[i] and 
                close[i] > sma[i] and close[i] > sma_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: ADX > 25, volume > 1.5x average, price < SMA, and price < 1-day 200 SMA
            elif (adx[i] > 25 and volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < sma[i] and close[i] < sma_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals