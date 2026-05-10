0

#!/usr/bin/env python3
"""
4h_ADX_Filtered_Donchian_Breakout_With_Volume
Hypothesis: Donchian breakout with ADX > 25 trend filter and volume spike works in both bull and bear markets.
ADX ensures we only trade in trending conditions, avoiding whipsaws in ranges. Volume confirms breakout strength.
Target: 25-40 trades/year by requiring ADX trend + Donchian breakout + volume confirmation.
"""

name = "4h_ADX_Filtered_Donchian_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        atr = np.full_like(high, np.nan)
        plus_dm_smooth = np.full_like(high, np.nan)
        minus_dm_smooth = np.full_like(high, np.nan)
        
        # First values (simple average)
        if len(high) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(high, np.nan)
        minus_di = np.full_like(high, np.nan)
        dx = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            if atr[i] != 0 and not np.isnan(atr[i]):
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX: smoothed DX
        adx = np.full_like(high, np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate ADX
    adx = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(df_4h['high'].values, df_4h['low'].values, 20)
    dc_upper_aligned = align_htf_to_ltf(prices, df_4h, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_4h, dc_lower)
    
    # Calculate 4h average volume for volume filter
    vol_avg_4h = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):  # 20-period MA
        vol_avg_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX (28), Donchian (20), volume avg (19)
    start_idx = max(28, 20, 19)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_avg_4h[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + trend + volume
            if close[i] > dc_upper_aligned[i] and strong_trend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + trend + volume
            elif close[i] < dc_lower_aligned[i] and strong_trend and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend weakens
            if close[i] < dc_lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend weakens
            if close[i] > dc_upper_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

0