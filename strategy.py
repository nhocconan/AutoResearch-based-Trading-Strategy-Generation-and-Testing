#!/usr/bin/env python3
"""
12h_1w_Trend_Filter_Volume_Spike
Hypothesis: 12h price above/below 1w EMA50 with volume spike captures strong trends in both bull and bear markets.
Long when price > 1w EMA50 + volume spike, short when price < 1w EMA50 + volume spike.
Exit when price crosses back below/above EMA50.
Volume filter reduces whipsaw in ranging markets. Target 15-30 trades/year per symbol.
"""

name = "12h_1w_Trend_Filter_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w EMA50 (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w = ema_50_1w
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        price = close[i]
        trend = trend_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price above 1w EMA50 + volume spike
            if price > trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price below 1w EMA50 + volume spike
            elif price < trend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 1w EMA50
            if price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 1w EMA50
            if price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals