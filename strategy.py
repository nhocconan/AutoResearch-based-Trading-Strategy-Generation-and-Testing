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
    
    # === Weekly high-low channel (primary signal) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Daily ADX for trend filter (28-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate directional movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr14 = np.full_like(close_1d, np.nan)
    plus_dm14 = np.full_like(close_1d, np.nan)
    minus_dm14 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 13:
            tr14[i] = np.sum(tr[i-13:i+1])
            plus_dm14[i] = np.sum(plus_dm[i-13:i+1])
            minus_dm14[i] = np.sum(minus_dm[i-13:i+1])
        elif i > 0:
            tr14[i] = np.sum(tr[1:i+1])
            plus_dm14[i] = np.sum(plus_dm[1:i+1])
            minus_dm14[i] = np.sum(minus_dm[1:i+1])
    
    # Calculate DI and DX
    plus_di = np.full_like(close_1d, np.nan)
    minus_di = np.full_like(close_1d, np.nan)
    dx = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if not np.isnan(tr14[i]) and tr14[i] > 0:
            plus_di[i] = 100 * plus_dm14[i] / tr14[i]
            minus_di[i] = 100 * minus_dm14[i] / tr14[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 27:
            valid_dx = dx[i-13:i+1]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) >= 14:
                adx[i] = np.mean(valid_dx)
        elif i >= 13:
            valid_dx = dx[1:i+1]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) >= 1:
                adx[i] = np.mean(valid_dx)
    
    # === Daily Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === Align indicators to daily timeframe ===
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    warmup = 100
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(high_1w_aligned[i]) or 
            np.isnan(low_1w_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: break above weekly high in strong trend (ADX > 25) with volume
            if close[i] > high_1w_aligned[i] and adx_aligned[i] > 25 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low in strong trend (ADX > 25) with volume
            elif close[i] < low_1w_aligned[i] and adx_aligned[i] > 25 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly low OR trend weakens (ADX < 20)
            if close[i] < low_1w_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly high OR trend weakens (ADX < 20)
            if close[i] > high_1w_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyChannel_ADX25_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0