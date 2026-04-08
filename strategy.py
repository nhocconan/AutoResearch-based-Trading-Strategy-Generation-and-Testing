#!/usr/bin/env python3
# 6h_1d_ema_pullback_v1
# Hypothesis: 6h trend following using 1d EMA(20) as dynamic trend filter and 6h EMA(8/21) for entry timing.
# In bull markets: price > 1d EMA20 + 6h EMA8 crosses above EMA21 → long
# In bear markets: price < 1d EMA20 + 6h EMA8 crosses below EMA21 → short
# Uses 1d trend filter to avoid counter-trend trades, reducing whipsaws in both bull and bear markets.
# Entry on EMA crossover provides timely signals while 1d EMA ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ema_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(20) - trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 6h EMA(8) and EMA(21) for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        ema8 = ema_8[i]
        ema21 = ema_21[i]
        price = close[i]
        ema20_1d = ema_20_1d_aligned[i]
        
        if np.isnan(ema8) or np.isnan(ema21) or np.isnan(ema20_1d):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if ema8 <= ema21 or price < ema20_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if ema8 >= ema21 or price > ema20_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if ema8 > ema21 and price > ema20_1d:
                position = 1
                signals[i] = 0.25
            elif ema8 < ema21 and price < ema20_1d:
                position = -1
                signals[i] = -0.25
    
    return signals