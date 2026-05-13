#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian channel (20) breakouts with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above upper channel with 1d uptrend and volume spike = long.
Breakdown below lower channel with 1d downtrend and volume spike = short.
Exit on opposite channel touch or trend reversal. Uses volume spike > 2x average and 1d trend as filter.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
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
    
    # Donchian Channel: 20-period high/low
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        upper = highest_high[i]
        lower = lowest_low[i]
        uptrend_1d_al = uptrend_1d_aligned[i]
        downtrend_1d_al = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above upper channel, 1d uptrend, volume confirmation
            if close[i] > upper and uptrend_1d_al and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower channel, 1d downtrend, volume confirmation
            elif close[i] < lower and downtrend_1d_al and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch lower channel or 1d trend turns down
            if close[i] < lower or not uptrend_1d_al:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch upper channel or 1d trend turns up
            if close[i] > upper or not downtrend_1d_al:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals