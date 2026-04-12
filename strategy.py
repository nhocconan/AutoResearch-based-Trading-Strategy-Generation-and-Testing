#!/usr/bin/env python3
"""
1d_1w_Momentum_Momentum_v1
Hypothesis: Combines 1d momentum (14-day ROC) with 1w momentum (4-week ROC) to capture strong trends. 
Long when both ROCs > 0 and price above 1d EMA(50), short when both ROCs < 0 and price below 1d EMA(50).
Uses 1w ADX > 25 to filter for trending markets only. Works in bull/bear by capturing momentum in trending conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Momentum_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for momentum and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ROC(4) - 4-week rate of change
    roc_4 = np.zeros_like(close_1w)
    roc_4[4:] = (close_1w[4:] - close_1w[:-4]) / close_1w[:-4] * 100
    
    # 1w ADX for trend filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(50) for trend confirmation
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to daily
    roc_4_aligned = align_htf_to_ltf(prices, df_1w, roc_4)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(roc_4_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_1w_aligned[i] > 25
        
        # Momentum signals
        long_signal = roc_4_aligned[i] > 0 and close[i] > ema_50[i] and trending
        short_signal = roc_4_aligned[i] < 0 and close[i] < ema_50[i] and trending
        
        # Exit: momentum reversal or trend weakening
        long_exit = roc_4_aligned[i] <= 0 or adx_1w_aligned[i] <= 25
        short_exit = roc_4_aligned[i] >= 0 or adx_1w_aligned[i] <= 25
        
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