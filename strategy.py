#!/usr/bin/env python3
name = "6h_Adaptive_Adx_Cci_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily ADX for trend strength (Higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Daily CCI for mean reversion signals
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (tp_1d - sma_tp) / (0.015 * mad)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # 6h Supertrend for entry/exit timing
    atr_6h = pd.Series(high - low).rolling(window=10, min_periods=10).mean().values
    upper = (high + low) / 2 + 3 * atr_6h
    lower = (high + low) / 2 - 3 * atr_6h
    
    supertrend = np.full(n, np.nan)
    dir = np.full(n, 1)
    
    for i in range(1, n):
        if np.isnan(upper[i-1]) or np.isnan(lower[i-1]):
            supertrend[i] = np.nan
            continue
            
        if close[i] > upper[i-1]:
            dir[i] = 1
        elif close[i] < lower[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if dir[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        supertrend[i] = lower[i] if dir[i] == 1 else upper[i]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(cci_aligned[i]) or 
            np.isnan(supertrend[i]) or np.isnan(dir[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: CCI < -100 (oversold) + uptrend + price above Supertrend
            if cci_aligned[i] < -100 and trend_filter and close[i] > supertrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 100 (overbought) + downtrend + price below Supertrend
            elif cci_aligned[i] > 100 and trend_filter and close[i] < supertrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI > 0 (mean reversion) or trend weakening (ADX < 20) or price below Supertrend
            if cci_aligned[i] > 0 or adx_aligned[i] < 20 or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI < 0 (mean reversion) or trend weakening (ADX < 20) or price above Supertrend
            if cci_aligned[i] < 0 or adx_aligned[i] < 20 or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals