#!/usr/bin/env python3
"""
6h_DMI_Crossover_Trend_Strength
Hypothesis: On 6h timeframe, use +DI/-DI crossover from DMI(14) as entry signal, 
filtered by ADX>25 for trend strength and 1d EMA50 for higher timeframe trend alignment.
Works in both bull and bear markets by capturing strong trending moves when ADX confirms strength.
Designed for low trade frequency (target 20-50 trades/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate True Range and Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # ATR and DI calculation
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        di_cross_up = plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]
        di_cross_down = minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]
        strong_trend = adx[i] > 25
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: DI bullish crossover + strong trend + 1d uptrend
            if di_cross_up and strong_trend and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: DI bearish crossover + strong trend + 1d downtrend
            elif di_cross_down and strong_trend and downtrend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: DI bearish crossover OR weak trend
            if di_cross_down or adx[i] < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: DI bullish crossover OR weak trend
            if di_cross_up or adx[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_DMI_Crossover_Trend_Strength"
timeframe = "6h"
leverage = 1.0