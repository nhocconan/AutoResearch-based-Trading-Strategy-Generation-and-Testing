#!/usr/bin/env python3
name = "4h_Donchian20_VolumeSpike_1dTrend_ADX"
timeframe = "4h"
leverage = 1.0

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
    
    # === DAILY TREND FILTER: EMA34 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === DAILY ADX (14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def wilder_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr14 = wilder_smoothing(tr, 14)
    plus_dm14 = wilder_smoothing(plus_dm, 14)
    minus_dm14 = wilder_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(tr14 > 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 > 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        adx[13] = np.nanmean(dx[:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H DONCHIAN CHANNEL (20) ===
    def donchian_channels(high_arr, low_arr, window):
        upper = np.full_like(high_arr, np.nan, dtype=float)
        lower = np.full_like(low_arr, np.nan, dtype=float)
        for i in range(window-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-window+1:i+1])
            lower[i] = np.min(low_arr[i-window+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # === VOLUME SPIKE (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for daily EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + above daily EMA34 + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + below daily EMA34 + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR below daily EMA34
            if close[i] < donchian_lower[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR above daily EMA34
            if close[i] > donchian_upper[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals