#!/usr/bin/env python3
name = "6h_12hVWAP_Bounce_1dTrend"
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
    
    # 12h VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    vwap_12h = (df_12h['close'] * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h = vwap_12h.values
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_12h_aligned[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: price near 12h VWAP (within 0.5%) and above 1d EMA50
            if abs(price - vwap) / vwap < 0.005 and price > ema50:
                signals[i] = 0.25
                position = 1
            # SHORT: price near 12h VWAP and below 1d EMA50
            elif abs(price - vwap) / vwap < 0.005 and price < ema50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price moves away from VWAP or trend changes
            if abs(price - vwap) / vwap > 0.015 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price moves away from VWAP or trend changes
            if abs(price - vwap) / vwap > 0.015 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals