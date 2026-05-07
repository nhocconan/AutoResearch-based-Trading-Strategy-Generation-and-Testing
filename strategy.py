#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume surge and ADX trend filter.
# Long when price breaks above 4h Donchian Upper(20) AND 1d volume surge (volume > 1.5 * 20 EMA) AND ADX > 25.
# Short when price breaks below 4h Donchian Lower(20) AND 1d volume surge AND ADX > 25.
# Uses daily volume for momentum confirmation and ADX to avoid ranging markets.
# Designed for fewer trades (target: 20-40/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by following 4h price action with volatility and trend filters.
name = "4h_Donchian20_1dVolumeSurge_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume surge and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume surge: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 1.5
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR and DM
    def smooth_signal(signal, period):
        smoothed = np.zeros_like(signal)
        smoothed[period-1] = np.mean(signal[:period])
        for i in range(period, len(signal)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + (signal[i] / period)
        return smoothed
    
    atr = smooth_signal(tr, 14)
    dm_plus_smooth = smooth_signal(dm_plus, 14)
    dm_minus_smooth = smooth_signal(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*period-1
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    adx[:27] = np.nan  # Not enough data for ADX
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_surge_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian Upper, volume surge, ADX > 25
            long_condition = (close[i] > donchian_upper_aligned[i]) and vol_surge_1d_aligned[i] and (adx_aligned[i] > 25)
            # Short condition: break below Donchian Lower, volume surge, ADX > 25
            short_condition = (close[i] < donchian_lower_aligned[i]) and vol_surge_1d_aligned[i] and (adx_aligned[i] > 25)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian Lower or ADX drops below 20
            if (close[i] < donchian_lower_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian Upper or ADX drops below 20
            if (close[i] > donchian_upper_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals