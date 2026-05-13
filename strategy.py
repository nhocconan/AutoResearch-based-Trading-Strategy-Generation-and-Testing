#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Trend_Volume
Hypothesis: Donchian channel (20) breakouts with 1d trend (EMA50) and volume confirmation work in both bull and bear markets.
Breakout above upper channel with uptrend and volume spike = long.
Breakdown below lower channel with downtrend and volume spike = short.
Exit on opposite channel touch or trend reversal. Uses 1w trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol.
"""

name = "12h_Donchian_Breakout_Trend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 20-period high/low
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close > ema_50
    downtrend_12h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        upper_ch = highest_20[i]
        lower_ch = lowest_20[i]
        uptrend = uptrend_12h[i]
        downtrend = downtrend_12h[i]
        uptrend_htf_1d = uptrend_1d_aligned[i]
        downtrend_htf_1d = downtrend_1d_aligned[i]
        uptrend_htf_1w = uptrend_1w_aligned[i]
        downtrend_htf_1w = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above upper channel, 12h uptrend, 1d/1w uptrend filter, volume confirmation
            if close[i] > upper_ch and uptrend and uptrend_htf_1d and uptrend_htf_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower channel, 12h downtrend, 1d/1w downtrend filter, volume confirmation
            elif close[i] < lower_ch and downtrend and downtrend_htf_1d and downtrend_htf_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch lower channel or 12h trend turns down
            if close[i] < lower_ch or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch upper channel or 12h trend turns up
            if close[i] > upper_ch or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals