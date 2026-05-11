#!/usr/bin/env python3
"""
6h_Weekly_Trend_With_Pullback_v1
Hypothesis: In trending markets, price often pulls back to the weekly trend before continuing.
Use weekly EMA50 as trend filter and daily ATR for pullback depth. Enter long when price
pulls back to weekly EMA in uptrend, short when price rallies to weekly EMA in downtrend.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Weekly_Trend_With_Pullback_v1"
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
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === DAILY DATA FOR ATR (PULLBACK DEPTH) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to weekly EMA in uptrend
            # Condition: close <= weekly EMA + 0.5*ATR and close > weekly EMA - 0.5*ATR
            # and weekly EMA is rising (trend up)
            ema_now = ema50_1w_aligned[i]
            ema_prev = ema50_1w_aligned[i-1]
            atr = atr14_1d_aligned[i]
            
            if ema_now > ema_prev and \
               close[i] <= ema_now + 0.5 * atr and \
               close[i] >= ema_now - 0.5 * atr:
                signals[i] = 0.25
                position = 1
            # Short: price rallies to weekly EMA in downtrend
            elif ema_now < ema_prev and \
                 close[i] >= ema_now - 0.5 * atr and \
                 close[i] <= ema_now + 0.5 * atr:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves significantly above weekly EMA
            if close[i] > ema50_1w_aligned[i] + 1.0 * atr14_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price moves significantly below weekly EMA
            if close[i] < ema50_1w_aligned[i] - 1.0 * atr14_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals