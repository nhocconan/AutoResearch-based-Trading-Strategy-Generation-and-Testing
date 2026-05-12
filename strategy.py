#!/usr/bin/env python3
# 1D_KAMA_DIRECTION_1W_TREND_FILTER
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) on daily timeframe captures trend direction,
# while weekly trend filter (EMA34) avoids counter-trend trades. KAMA adapts to market noise,
# reducing whipsaws in ranging markets and capturing trends in trending markets.
# Works in bull markets (follows upward KAMA slope) and bear markets (follows downward slope).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1D_KAMA_DIRECTION_1W_TREND_FILTER"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (2, 10, 30)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Efficiency ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Vectorized ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            ch = np.abs(close_1d[i] - close_1d[i-10])
            vol = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for KAMA calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(kama_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA and weekly trend up
            if (close[i] > kama_aligned[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and weekly trend down
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below KAMA or weekly trend turns down
            if (close[i] < kama_aligned[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above KAMA or weekly trend turns up
            if (close[i] > kama_aligned[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals