#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_50EMA_Pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Supertrend on 6h: ATR(10), multiplier=3
    atr_period = 10
    atr_multiplier = 3.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(close)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic upper and lower bands
    basic_ub = (high + low) / 2 + atr_multiplier * atr
    basic_lb = (high + low) / 2 - atr_multiplier * atr
    
    # Final upper and lower bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close)
    supertrend[0] = final_ub[0]
    for i in range(1, len(close)):
        if close[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
        else:
            supertrend[i] = final_lb[i]
    
    # Align 1d EMA50 to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, atr_period)
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st = supertrend[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: price above Supertrend (uptrend) and above 1d EMA50
            if close[i] > st and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Enter short: price below Supertrend (downtrend) and below 1d EMA50
            elif close[i] < st and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Supertrend
            if close[i] < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Supertrend
            if close[i] > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals