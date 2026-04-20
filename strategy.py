#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate 6h Supertrend (ATR=10, multiplier=3)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr = np.zeros(n)
    atr[9] = np.mean(tr[:10])
    for i in range(10, n):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    
    # Initialize
    supertrend = np.full(n, np.nan)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i-1] > supertrend[i-1]:
            upper[i] = min(upper[i], upper[i-1])
        else:
            lower[i] = max(lower[i], lower[i-1])
        
        if close[i] > upper[i-1]:
            direction[i] = 1
        elif close[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower[i]
        else:
            supertrend[i] = upper[i]
    
    # Weekly ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1w = high_1w - low_1w
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    tr_1w[0] = tr1w[0]
    
    # ATR 1w
    atr_1w = np.zeros(len(high_1w))
    if len(high_1w) >= 14:
        atr_1w[13] = np.mean(tr_1w[:14])
        for i in range(14, len(high_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    if len(high_1w) >= 14:
        atr_1w_smooth = np.zeros(len(high_1w))
        plus_dm_smooth = np.zeros(len(high_1w))
        minus_dm_smooth = np.zeros(len(high_1w))
        
        atr_1w_smooth[13] = np.mean(atr_1w[:14])
        plus_dm_smooth[13] = np.mean(plus_dm[:14])
        minus_dm_smooth[13] = np.mean(minus_dm[:14])
        
        for i in range(14, len(high_1w)):
            atr_1w_smooth[i] = (atr_1w_smooth[i-1] * 13 + atr_1w[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
        
        # ADX calculation
        plus_di = 100 * plus_dm_smooth / atr_1w_smooth
        minus_di = 100 * minus_dm_smooth / atr_1w_smooth
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.zeros(len(high_1w))
        if len(high_1w) >= 28:
            adx[27] = np.mean(dx[14:28])
            for i in range(28, len(high_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    else:
        adx = np.zeros(len(high_1w))
    
    # Align weekly ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Generate signals
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if NaN values
        if np.isnan(supertrend[i]) or np.isnan(adx_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st_val = supertrend[i]
        adx_val = adx_6h[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Supertrend uptrend, ADX > 25, volume spike
            if direction[i] == 1 and adx_val > 25 and vol_val > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, ADX > 25, volume spike
            elif direction[i] == -1 and adx_val > 25 and vol_val > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Supertrend downtrend
            if direction[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Supertrend uptrend
            if direction[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Supertrend_ADX_Volume"
timeframe = "6h"
leverage = 1.0