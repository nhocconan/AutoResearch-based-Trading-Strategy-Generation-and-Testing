#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation
Hypothesis: On the 1h chart, use 4h and 1d for directional context. 
- Long when price breaks above daily R1 with 4h uptrend and volume confirmation
- Short when price breaks below daily S1 with 4h downtrend and volume confirmation
- Uses daily Camarilla levels (more reliable than intraday) and 4h trend filter to avoid counter-trend trades
- Volume filter reduces false breakouts
- Session filter (08-20 UTC) avoids low-liquidity hours
- Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag
"""

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = volumes = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla R1 and S1
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    diff_1d = high_1d - low_1d
    cam_r1 = close_1d + diff_1d * 1.1 / 12
    cam_s1 = close_1d - diff_1d * 1.1 / 12
    
    # Align Camarilla levels to 1h (wait for daily close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h trend: 21 EMA
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    uptrend_4h = close_4h > ema_21_4h
    downtrend_4h = close_4h < ema_21_4h
    
    # Align 4h trend to 1h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # Volume confirmation: volume > 1.3 * 20-period average (less strict to allow more signals)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Get aligned values for current bar
        r1 = cam_r1_aligned[i]
        s1 = cam_s1_aligned[i]
        uptrend = uptrend_4h_aligned[i]
        downtrend = downtrend_4h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above daily R1, 4h uptrend, volume confirmation
            if close[i] > r1 and uptrend and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below daily S1, 4h downtrend, volume confirmation
            elif close[i] < s1 and downtrend and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below daily S1 or 4h trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above daily R1 or 4h trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals