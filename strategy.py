#!/usr/bin/env python3
"""
Hypothesis: 12-hour strategy using 1-day KAMA trend direction filtered by 1-week volatility regime.
Long when 1-day KAMA is rising and price > KAMA during low weekly volatility with volume confirmation.
Short when 1-day KAMA is falling and price < KAMA during low weekly volatility with volume confirmation.
Exit when price crosses KAMA in opposite direction or volatility expands significantly.
Designed for low turnover: ~10-20 trades/year per symbol to minimize fee drift.
"""
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
    
    # Load 1-day data once for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day KAMA parameters
    close_1d = df_1d['close'].values
    er = np.zeros(len(close_1d))
    sc = np.zeros(len(close_1d))
    kama = np.zeros(len(close_1d))
    
    # Efficiency Ratio calculation
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    for i in range(10, len(close_1d)):
        if i >= 10:
            dir_change = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = dir_change / volatility if volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # KAMA calculation
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1-day KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=0)
    
    # Load 1-week data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index (2 bars per day: 24/12 = 2)
        idx_1d = i // 2
        if idx_1d < 10:  # need enough for KAMA calculation
            continue
        
        # Get previous 1-day values to avoid look-ahead
        kama_prev = kama[idx_1d - 1] if idx_1d - 1 < len(kama) else kama[-1]
        kama_slope_prev = kama_slope[idx_1d - 1] if idx_1d - 1 < len(kama_slope) else kama_slope[-1]
        if np.isnan(kama_prev) or np.isnan(kama_slope_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        kama_arr = np.full(len(df_1d), kama_prev)
        kama_slope_arr = np.full(len(df_1d), kama_slope_prev)
        kama_12h = align_htf_to_ltf(prices, df_1d, kama_arr)[i]
        kama_slope_12h = align_htf_to_ltf(prices, df_1d, kama_slope_arr)[i]
        
        # 1-week index (14 bars per week: 7*24/12 = 14)
        idx_1w = i // 14
        if idx_1w < 20:  # need enough for ATR MA
            continue
        
        # Get previous 1-week ATR and MA to avoid look-ahead
        atr_prev = atr_1w[idx_1w - 1] if idx_1w - 1 < len(atr_1w) else atr_1w[-1]
        atr_ma_prev = atr_ma_1w[idx_1w - 1] if idx_1w - 1 < len(atr_ma_1w) else atr_ma_1w[-1]
        if np.isnan(atr_prev) or np.isnan(atr_ma_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        atr_arr = np.full(len(df_1w), atr_prev)
        atr_ma_arr = np.full(len(df_1w), atr_ma_prev)
        atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_arr)[i]
        atr_ma_1w_12h = align_htf_to_ltf(prices, df_1w, atr_ma_arr)[i]
        
        if position == 0:
            # Long: KAMA rising + price > KAMA + low volatility + volume surge
            if (kama_slope_12h > 0 and 
                close[i] > kama_12h and 
                atr_1w_12h < atr_ma_1w_12h and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: KAMA falling + price < KAMA + low volatility + volume surge
            elif (kama_slope_12h < 0 and 
                  close[i] < kama_12h and 
                  atr_1w_12h < atr_ma_1w_12h and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price below KAMA or volatility expansion
            if close[i] < kama_12h or atr_1w_12h > atr_ma_1w_12h * 1.5:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price above KAMA or volatility expansion
            if close[i] > kama_12h or atr_1w_12h > atr_ma_1w_12h * 1.5:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_KAMA_1wVol_Volume"
timeframe = "12h"
leverage = 1.0