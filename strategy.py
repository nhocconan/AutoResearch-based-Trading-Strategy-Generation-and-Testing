#!/usr/bin/env python3
"""
1d_4hTrend_WeeklyTrend_VolumeBreakout
Hypothesis: Daily breakouts above/below 20-period Donchian channels with 4h trend alignment and weekly trend filter work in both bull and bear markets.
Breakout above upper band with 4h uptrend and weekly uptrend = long.
Breakdown below lower band with 4h downtrend and weekly downtrend = short.
Exit on opposite band touch. Uses volume confirmation to filter false breakouts.
Target: 10-25 trades/year per symbol.
"""

name = "1d_4hTrend_WeeklyTrend_VolumeBreakout"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period high/low)
    highest_20 = np.zeros(n)
    lowest_20 = np.zeros(n)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # 4h trend: EMA50
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50_4h
    downtrend_4h = close < ema_50_4h
    
    # Weekly trend: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        upper = highest_20[i]
        lower = lowest_20[i]
        uptrend_4h_val = uptrend_4h[i]
        downtrend_4h_val = downtrend_4h[i]
        uptrend_1w_val = uptrend_1w_aligned[i]
        downtrend_1w_val = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above upper band, 4h uptrend, weekly uptrend, volume confirmation
            if close[i] > upper and uptrend_4h_val and uptrend_1w_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower band, 4h downtrend, weekly downtrend, volume confirmation
            elif close[i] < lower and downtrend_4h_val and downtrend_1w_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch lower band
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch upper band
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals