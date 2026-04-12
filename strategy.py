#!/usr/bin/env python3
"""
4h_1d_Keltner_Reversal_v1
Hypothesis: Use 1d Keltner Channels with 10-period ATR for mean-reversion signals. Buy near lower band (oversold) and sell near upper band (overbought) with 4h ADX<20 to ensure ranging markets. Works in both bull/bear by capturing mean-reversion in ranging conditions while avoiding strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Keltner_Reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h ADX for regime filter (range detection)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Daily Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(10) for Keltner Channel
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_10 = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # EMA(20) of close
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10
    
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: only trade when ADX < 20 (ranging market)
        ranging = adx_4h_aligned[i] < 20
        
        # Mean-reversion signals
        long_signal = close[i] <= keltner_lower_aligned[i] and ranging
        short_signal = close[i] >= keltner_upper_aligned[i] and ranging
        
        # Exit: return to EMA(20)
        ema_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
        long_exit = close[i] >= ema_aligned[i]
        short_exit = close[i] <= ema_aligned[i]
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals