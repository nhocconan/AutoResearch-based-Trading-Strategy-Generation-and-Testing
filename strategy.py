#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1h price breaking above 4h Camarilla R1 in 4h uptrend or below 4h S1 in 4h downtrend continues with momentum.
# Volume confirmation from 1d filters false breakouts. Session filter (08-20 UTC) reduces noise.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends) by only trading in direction of 4h trend.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (standard formula)
    # P = (H + L + C) / 3
    # R1 = P + (H - L) * 1.1 / 2
    # S1 = P - (H - L) * 1.1 / 2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    r1_4h = pivot_4h + (high_4h - low_4h) * 1.1 / 2
    s1_4h = pivot_4h - (high_4h - low_4h) * 1.1 / 2
    
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate 4h trend (EMA34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d volume MA for confirmation
    volume_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA34 (34), 4h Camarilla (10), 1d volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # 4h trend filter
        uptrend_4h = close[i] > ema_34_4h_aligned[i]
        downtrend_4h = close[i] < ema_34_4h_aligned[i]
        
        # 1d volume confirmation (current volume > 20-period EMA)
        volume_confirm = volume[i] > volume_ma_1d_aligned[i]
        
        if position == 0 and in_session:
            # Long entry: 4h uptrend + price breaks above 4h R1 + volume
            if uptrend_4h and close[i] > r1_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend + price breaks below 4h S1 + volume
            elif downtrend_4h and close[i] < s1_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or session ends
            if not uptrend_4h or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or session ends
            if not downtrend_4h or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals