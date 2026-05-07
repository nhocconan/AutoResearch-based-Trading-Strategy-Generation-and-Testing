#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend"
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
    
    # Load daily data ONCE for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = close + (range_hl * 1.1 / 12)
    s1 = close - (range_hl * 1.1 / 12)
    
    # Align to 12h timeframe (will use previous day's levels)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be ready
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of daily EMA34
        daily_uptrend = close[i] > ema_34_12h[i]
        daily_downtrend = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + daily uptrend
            if close[i] > r1_12h[i] and volume_spike[i] and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + daily downtrend
            elif close[i] < s1_12h[i] and volume_spike[i] and daily_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below R1 or trend reverses
            if close[i] < r1_12h[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above S1 or trend reverses
            if close[i] > s1_12h[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals