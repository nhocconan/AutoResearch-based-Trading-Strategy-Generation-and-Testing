#!/usr/bin/env python3
"""
4h_1d_RCI_Cross_With_Volume_Filter_v1
Hypothesis: Use 1d Relative Strength Index (RSI) crossover with volume confirmation and 4h ADX trend filter.
Buy when RSI crosses above 30 from below in ranging/weak trend markets (ADX<25), sell when RSI crosses below 70 from above.
This captures mean-reversion in ranging markets while avoiding strong trends, working in both bull/bear markets.
Volume filter ensures only high-confidence signals. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RCI_Cross_With_Volume_Filter_v1"
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
    
    # 4h ADX for trend filter
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
    
    # Daily RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI crossover signals
    rsi_above_30 = rsi > 30
    rsi_below_30 = rsi < 30
    rsi_above_70 = rsi > 70
    rsi_below_70 = rsi < 70
    
    # Cross above 30: previous below, current above
    rsi_cross_above_30 = rsi_above_30 & np.roll(rsi_below_30, 1)
    # Cross below 70: previous above, current below
    rsi_cross_below_70 = rsi_below_70 & np.roll(rsi_above_70, 1)
    
    # Handle first element
    rsi_cross_above_30[0] = False
    rsi_cross_below_70[0] = False
    
    rsi_cross_above_30_aligned = align_htf_to_ltf(prices, df_1d, rsi_cross_above_30.astype(float))
    rsi_cross_below_70_aligned = align_htf_to_ltf(prices, df_1d, rsi_cross_below_70.astype(float))
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(rsi_cross_above_30_aligned[i]) or 
            np.isnan(rsi_cross_below_70_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade when ADX < 25 (weak trend/ranging)
        weak_trend = adx_4h_aligned[i] < 25
        
        # Entry signals
        long_signal = rsi_cross_above_30_aligned[i] > 0.5 and weak_trend and volume_filter[i]
        short_signal = rsi_cross_below_70_aligned[i] > 0.5 and weak_trend and volume_filter[i]
        
        # Exit: opposite RSI crossover
        long_exit = rsi_cross_below_70_aligned[i] > 0.5
        short_exit = rsi_cross_above_30_aligned[i] > 0.5
        
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