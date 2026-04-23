#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with 1w ADX regime filter and volume spike confirmation.
Long when KAMA rising AND 1w ADX > 25 (trending) AND volume > 2.0x 20-period average.
Short when KAMA falling AND 1w ADX > 25 (trending) AND volume > 2.0x 20-period average.
Exit when KAMA reverses direction OR volume drops below average OR ADX < 20 (range).
KAMA adapts to market noise, reducing false signals in choppy markets.
1w ADX ensures we only trade in strong trending regimes across multiple timeframes.
Volume confirmation avoids low-conviction breakouts.
Designed for 1d timeframe targeting 30-100 total trades over 4 years to minimize fee drag.
Works in both bull and bear markets by only taking trades in strong trends.
"""

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
    
    # Load 1d data for KAMA calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 1d data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period sum of absolute changes
    # Pad the beginning with NaN for alignment
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 1d timeframe (identity alignment since primary is 1d)
    kama_aligned = kama  # Already on 1d timeframe
    
    # Calculate KAMA direction (1 if rising, -1 if falling, 0 if flat)
    kama_diff = np.diff(kama_aligned, prepend=kama_aligned[0])
    kama_dir = np.where(kama_diff > 0, 1, np.where(kama_diff < 0, -1, 0))
    
    # Load 1w data for ADX regime filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w data (14-period)
    # True Range
    tr1 = np.subtract(high_1w, low_1w)
    tr2 = np.subtract(np.abs(high_1w), np.abs(np.concatenate([[close_1w[0]], close_1w[:-1]])))
    tr3 = np.subtract(np.abs(low_1w), np.abs(np.concatenate([[close_1w[0]], close_1w[:-1]])))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    # Pad first element
    tr = np.concatenate([[np.nan], tr[1:]])
    
    # Directional Movement
    up_move = np.subtract(high_1w, np.concatenate([[high_1w[0]], high_1w[:-1]]))
    down_move = np.subtract(np.concatenate([[low_1w[0]], low_1w[:-1]]), low_1w)
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus = np.full_like(up_move, np.nan)
    dm_minus = np.full_like(down_move, np.nan)
    
    # Initial values (simple average)
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus[tr_period-1] = np.nanmean(up_move[1:tr_period])
        dm_minus[tr_period-1] = np.nanmean(down_move[1:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        if not np.isnan(atr[i-1]):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            dm_plus[i] = (dm_plus[i-1] * (tr_period - 1) + up_move[i]) / tr_period
            dm_minus[i] = (dm_minus[i-1] * (tr_period - 1) + down_move[i]) / tr_period
    
    # Directional Indicators
    plus_di = np.full_like(atr, np.nan)
    minus_di = np.full_like(atr, np.nan)
    dx = np.full_like(atr, np.nan)
    
    for i in range(tr_period, len(atr)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = 100 * (dm_plus[i] / atr[i])
            minus_di[i] = 100 * (dm_minus[i] / atr[i])
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    if len(dx) >= adx_period * 2:
        adx[adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
        for i in range(2*adx_period, len(dx)):
            if not np.isnan(dx[i-1]):
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        kama_dir_val = kama_dir[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: KAMA rising AND ADX > 25 (trending) AND volume spike
            if (kama_dir_val > 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND ADX > 25 (trending) AND volume spike
            elif (kama_dir_val < 0 and adx_val > 25 and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA reverses OR volume drops OR ADX < 20 (range)
                if (kama_dir_val <= 0 or vol_current < vol_ma_val or adx_val < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA reverses OR volume drops OR ADX < 20 (range)
                if (kama_dir_val >= 0 or vol_current < vol_ma_val or adx_val < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_KAMA_1wADX_Volume"
timeframe = "1d"
leverage = 1.0