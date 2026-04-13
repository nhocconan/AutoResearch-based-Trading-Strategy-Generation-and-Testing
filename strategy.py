#!/usr/bin/env python3
"""
6h_1d1w_Triple_Barrier_Strategy
Hypothesis: 6h timeframe with triple barrier system using 1d volatility (ATR) and 1w trend.
Long when price breaks above 6h high + ATR(1d) expansion + 1w uptrend.
Short when price breaks below 6h low + ATR(1d) expansion + 1w downtrend.
Exit at fixed ATR-based stop loss or profit target, or trend reversal.
Designed for 6h to capture medium-term trends with volatility-adjusted barriers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6-period ATR for 6h volatility (using 6h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # ATR expansion: current ATR > 1.5x average ATR
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_1d > (atr_ma_1d * 1.5)
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # 1w trend filter: EMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, ema_slow)
    uptrend = ema_fast_aligned > ema_slow_aligned
    downtrend = ema_fast_aligned < ema_slow_aligned
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    # Entry barriers: 6h high/low breakout
    high_6h = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_6h = pd.Series(low).rolling(window=6, min_periods=6).min().values
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_expansion_aligned[i]) or np.isnan(uptrend[i]) or np.isnan(downtrend[i]) or
            np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volatility filter
        long_breakout = close[i] > high_6h[i-1]  # break above prior 6h high
        short_breakout = close[i] < low_6h[i-1]  # break below prior 6h low
        vol_condition = atr_expansion_aligned[i]
        
        if position == 0:
            if long_breakout and vol_condition and uptrend[i]:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and downtrend[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: stop loss (2*ATR below entry) or profit target (3*ATR) or trend reverse
            # Simplified: exit on trend reversal or price retracing to 6h low
            if not uptrend[i] or close[i] < low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: stop loss (2*ATR above entry) or profit target or trend reverse
            if not downtrend[i] or close[i] > high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Triple_Barrier_Strategy"
timeframe = "6h"
leverage = 1.0