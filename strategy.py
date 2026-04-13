#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX trend filter.
# Donchian breakouts capture breakout moves, volume confirms institutional participation,
# ADX ensures trades occur in trending markets (avoiding chop). Works in bull/bear via
# directional breakouts. Target: 15-25 trades per year (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    dm_plus = np.zeros(len(high_1d))
    dm_minus = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    def smooth_series(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.sum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full(len(high_1d), np.nan)
    di_minus = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if atr[i] > 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
    
    # DX and ADX
    dx = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if di_plus[i] + di_minus[i] > 0:
            dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    adx = smooth_series(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_now = vol_1d[i // 24] if i // 24 < len(vol_1d) else vol_1d[-1]
        avg_vol_1d_now = avg_vol_1d_aligned[i]
        adx_now = adx_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirm = vol_1d_now > 1.5 * avg_vol_1d_now if not np.isnan(avg_vol_1d_now) else False
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_now > 25
        
        if position == 0:
            # Long breakout: price > Donchian high + volume + ADX
            if (close[i] > donchian_high[i] and volume_confirm and adx_filter):
                position = 1
                signals[i] = position_size
            # Short breakout: price < Donchian low + volume + ADX
            elif (close[i] < donchian_low[i] and volume_confirm and adx_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < Donchian low (breakdown) or ADX weakening
            if close[i] < donchian_low[i] or adx_now < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > Donchian high (breakout) or ADX weakening
            if close[i] > donchian_high[i] or adx_now < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0