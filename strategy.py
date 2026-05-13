#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Breakout_Volume
Hypothesis: Camarilla pivot breakouts with 4h/1d trend alignment and volume confirmation work in both bull and bear markets.
Breakout above R3 with 4h/1d uptrend and volume spike = long.
Breakdown below S3 with 4h/1d downtrend and volume spike = short.
Exit on opposite pivot level (R2/S2) or trend reversal. Uses 4h for signal direction, 1h for entry timing.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

name = "1h_4h1d_Camarilla_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 4h OHLC for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 4h bar's H/L/C
    # R4 = C + ((H-L) * 1.5), R3 = C + ((H-L) * 1.25), R2 = C + ((H-L) * 1.166)
    # S2 = C - ((H-L) * 1.166), S3 = C - ((H-L) * 1.25), S4 = C - ((H-L) * 1.5)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to use previous bar's data (no look-ahead)
    h1 = high_4h[:-1]
    l1 = low_4h[:-1]
    c1 = close_4h[:-1]
    
    # Calculate levels
    r4 = c1 + (h1 - l1) * 1.5
    r3 = c1 + (h1 - l1) * 1.25
    r2 = c1 + (h1 - l1) * 1.166
    s2 = c1 - (h1 - l1) * 1.166
    s3 = c1 - (h1 - l1) * 1.25
    s4 = c1 - (h1 - l1) * 1.5
    
    # Align levels to 1h timeframe (already shifted by 1 in calculation)
    r4_1h = align_htf_to_ltf(prices, df_4h, r4)
    r3_1h = align_htf_to_ltf(prices, df_4h, r3)
    r2_1h = align_htf_to_ltf(prices, df_4h, r2)
    s2_1h = align_htf_to_ltf(prices, df_4h, s2)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3)
    s4_1h = align_htf_to_ltf(prices, df_4h, s4)
    
    # 4h trend: EMA50 on 4h close
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # 1d trend filter: EMA50 on 1d close
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
        r3 = r3_1h[i]
        s3 = s3_1h[i]
        r2 = r2_1h[i]
        s2 = s2_1h[i]
        uptrend_4h = uptrend_4h_aligned[i]
        downtrend_4h = downtrend_4h_aligned[i]
        uptrend_1d = uptrend_1d_aligned[i]
        downtrend_1d = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3, 4h/1d uptrend, volume confirmation
            if close[i] > r3 and uptrend_4h and uptrend_1d and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S3, 4h/1d downtrend, volume confirmation
            elif close[i] < s3 and downtrend_4h and downtrend_1d and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch R2 or 4h/1d trend turns down
            if close[i] >= r2 or not (uptrend_4h and uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch S2 or 4h/1d trend turns up
            if close[i] <= s2 or not (downtrend_4h and downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals