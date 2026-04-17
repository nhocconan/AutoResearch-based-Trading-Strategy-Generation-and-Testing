#!/usr/bin/env python3
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
    
    # === 1d Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian upper and lower (20-period)
    donch_up = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_up[i] = np.max(high_1d[i-20:i+1])
        donch_low[i] = np.min(low_1d[i-20:i+1])
    
    # === 1d ADX (14) for trend strength ===
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - high_1d[i-1]), 
                    abs(low_1d[i] - low_1d[i-1]))
    
    # Wilder's smoothing for TR, +DM, -DM
    atr = np.zeros(len(tr))
    plus_dm_smooth = np.zeros(len(plus_dm))
    minus_dm_smooth = np.zeros(len(minus_dm))
    
    # Initial values (first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:15])
        plus_dm_smooth[13] = np.mean(plus_dm[1:15])
        minus_dm_smooth[13] = np.mean(minus_dm[1:15])
        
        # Smooth subsequent values
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.zeros(len(high_1d))
    minus_di = np.zeros(len(high_1d))
    dx = np.zeros(len(high_1d))
    
    for i in range(14, len(high_1d)):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX: smoothed DX
    adx = np.zeros(len(dx))
    if len(dx) >= 28:  # Need 14 for DX + 14 for smoothing
        adx[27] = np.mean(dx[14:28])
        for i in range(28, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # === 1d Volume MA (20) ===
    vol_ma_20_1d = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20_1d[i] = np.mean(volume_1d[max(0, i-9):i+1]) if i > 0 else volume_1d[0]
    
    # Align to 12h timeframe
    donch_up_aligned = align_htf_to_ltf(prices, df_1d, donch_up)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_up_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.8
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian upper with volume and trend
            if close[i] > donch_up_aligned[i] and vol_confirm and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian lower with volume and trend
            elif close[i] < donch_low_aligned[i] and vol_confirm and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: opposite Donchian break or loss of momentum
        elif position == 1:
            # Exit long: price breaks below Donchian lower
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper
            if close[i] > donch_up_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_ADX_Volume"
timeframe = "12h"
leverage = 1.0