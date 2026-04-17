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
    
    # === 1d Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels
    upper_20 = np.full_like(high_1d, np.nan)
    lower_20 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            upper_20[i] = np.max(high_1d[i-19:i+1])
            lower_20[i] = np.min(low_1d[i-19:i+1])
        else:
            upper_20[i] = np.nan
            lower_20[i] = np.nan
    
    # === 1d ATR (14) for volatility filter ===
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close[:-1]) if len(close) > 1 else np.array([])
    tr3 = np.abs(low_1d[1:] - close[:-1]) if len(close) > 1 else np.array([])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))]) if len(tr1) > 0 else np.array([np.nan])
    
    atr_14 = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 13:
            atr_14[i] = np.mean(tr[i-13:i+1])
        elif i > 0:
            atr_14[i] = np.mean(tr[1:i+1]) if i >= 1 else np.nan
        else:
            atr_14[i] = np.nan
    
    # === 1d ADX (14) for trend strength ===
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr_14 = np.full_like(high_1d, np.nan)
    plus_dm_14 = np.full_like(high_1d, np.nan)
    minus_dm_14 = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 13:
            tr_14[i] = np.sum(tr[i-13:i+1])
            plus_dm_14[i] = np.sum(plus_dm[i-13:i+1])
            minus_dm_14[i] = np.sum(minus_dm[i-13:i+1])
        elif i > 0:
            tr_14[i] = np.sum(tr[1:i+1])
            plus_dm_14[i] = np.sum(plus_dm[1:i+1])
            minus_dm_14[i] = np.sum(minus_dm[1:i+1])
        else:
            tr_14[i] = np.nan
            plus_dm_14[i] = np.nan
            minus_dm_14[i] = np.nan
    
    # Avoid division by zero
    plus_di = np.full_like(high_1d, np.nan)
    minus_di = np.full_like(high_1d, np.nan)
    dx = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tr_14[i]) and tr_14[i] > 0:
            plus_di[i] = 100 * plus_dm_14[i] / tr_14[i]
            minus_di[i] = 100 * minus_dm_14[i] / tr_14[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0.0
        else:
            plus_di[i] = np.nan
            minus_di[i] = np.nan
            dx[i] = np.nan
    
    adx = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 27:  # Need 14 for DX + 14 for smoothing
            valid_dx = dx[i-13:i+1]
            if not np.any(np.isnan(valid_dx)):
                adx[i] = np.mean(valid_dx)
        elif i > 0:
            adx[i] = np.nan
        else:
            adx[i] = np.nan
    
    # === Align indicators to 4h timeframe ===
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Volume confirmation ===
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
    warmup = 100
    
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above upper Donchian in strong trend (ADX > 25) with volume
            if (close[i] > upper_20_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower Donchian in strong trend (ADX > 25) with volume
            elif (close[i] < lower_20_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to lower Donchian OR trend weakens (ADX < 20)
            if (close[i] < lower_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper Donchian OR trend weakens (ADX < 20)
            if (close[i] > upper_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX25_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0