#!/usr/bin/env python3
"""
1h_Donchian_Breakout_Trend_Volume
Hypothesis: 1-hour Donchian breakouts with 4h trend and volume confirmation work in both bull and bear markets.
Breakout above 20-period high with 4h uptrend and volume spike = long.
Breakdown below 20-period low with 4h downtrend and volume spike = short.
Exit on opposite band touch or trend reversal.
Uses 1d trend filter for higher timeframe bias. Uses 4h EMA50 for trend.
Target: 15-37 trades/year per session (60-150 total over 4 years).
"""

name = "1h_Donchian_Breakout_Trend_Volume"
timeframe = "1h"
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
    high_20 = np.zeros(n)
    low_20 = np.zeros(n)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Get values
        upper = high_20[i]
        lower = low_20[i]
        uptrend_4h_val = uptrend_4h[i]
        downtrend_4h_val = downtrend_4h[i]
        uptrend_1d_val = uptrend_1d[i]
        downtrend_1d_val = downtrend_1d[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above 20-period high, 4h uptrend, 1d uptrend, volume confirmation
            if close[i] > upper and uptrend_4h_val and uptrend_1d_val and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: break below 20-period low, 4h downtrend, 1d downtrend, volume confirmation
            elif close[i] < lower and downtrend_4h_val and downtrend_1d_val and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch 20-period low or 4h trend turns down
            if close[i] < lower or not uptrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch 20-period high or 4h trend turns up
            if close[i] > upper or not downtrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals