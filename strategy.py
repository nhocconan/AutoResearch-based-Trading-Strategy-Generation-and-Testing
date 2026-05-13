#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_1dTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Calculate previous day's Camarilla levels (R1, S1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.2917 * camarilla_range
    s1 = prev_close - 0.2917 * camarilla_range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1[i]) or 
            np.isnan(s1[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during active session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 + 1d uptrend
            if close[i] > r1[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 + 1d downtrend
            elif close[i] < s1[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend reverses
            if close[i] < s1[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend reverses
            if close[i] > r1[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals