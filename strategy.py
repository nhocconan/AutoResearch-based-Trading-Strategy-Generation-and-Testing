#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Trend_Volume
Hypothesis: Camarilla pivot points (R1/S1) on 4h with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R1 with 4h uptrend and volume spike = long.
Breakdown below S1 with 4h downtrend and volume spike = short.
Exit on opposite touch (S1 for long, R1 for short). Uses session filter (08-20 UTC) to reduce noise.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
"""

name = "1h_Camarilla_R1S1_Breakout_Trend_Volume"
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
    
    # 4h OHLC for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    # Avoid look-ahead: only use previous bar's data
    R1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    S1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align to 1h (wait for 4h bar to close)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # 4h trend: EMA34 on close
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_4h = df_4h['close'].values > ema_34_4h
    downtrend_4h = df_4h['close'].values < ema_34_4h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 24-period average
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        r1 = R1_4h_aligned[i]
        s1 = S1_4h_aligned[i]
        uptrend_4h_val = uptrend_4h_aligned[i]
        downtrend_4h_val = downtrend_4h_aligned[i]
        uptrend_1d_val = uptrend_1d_aligned[i]
        downtrend_1d_val = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 4h uptrend, 1d uptrend filter, volume confirmation
            if close[i] > r1 and uptrend_4h_val and uptrend_1d_val and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S1, 4h downtrend, 1d downtrend filter, volume confirmation
            elif close[i] < s1 and downtrend_4h_val and downtrend_1d_val and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 4h trend turns down
            if close[i] < s1 or not uptrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch R1 or 4h trend turns up
            if close[i] > r1 or not downtrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals