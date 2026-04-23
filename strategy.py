#!/usr/bin/env python3
"""
Hypothesis: 1h 4h EMA crossover with 1d ADX trend filter and volume confirmation for 1h timeframe.
Long when 4h EMA20 > EMA50 AND 1d ADX > 25 AND volume > 1.5x 20-period average AND price > 4h EMA20.
Short when 4h EMA20 < EMA50 AND 1d ADX > 25 AND volume > 1.5x 20-period average AND price < 4h EMA20.
Exit when 4h EMA crossover reverses OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.20) targeting 15-37 trades/year on 1h timeframe.
4h EMA crossover provides trend direction, 1d ADX filters ranging markets, volume confirmation ensures momentum.
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
    
    # Calculate 4h EMA20 and EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema20_val = ema20_4h_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: 4h EMA20 > EMA50 AND 1d ADX > 25 AND volume > 1.5x avg AND price > EMA20
            if ema20_val > ema50_val and adx_val > 25 and volume[i] > 1.5 * vol_ma_val and price > ema20_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: 4h EMA20 < EMA50 AND 1d ADX > 25 AND volume > 1.5x avg AND price < EMA20
            elif ema20_val < ema50_val and adx_val > 25 and volume[i] > 1.5 * vol_ma_val and price < ema20_val:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: 4h EMA crossover reverses
            if position == 1 and ema20_val < ema50_val:
                exit_signal = True
            elif position == -1 and ema20_val > ema50_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hEMA20_50_Crossover_1dADX25_VolumeConfirmation_EMACrossoverExit_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0