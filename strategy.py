#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
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
    
    # 4h trend filter: EMA50 (used as trend direction)
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Daily high/low for Camarilla levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each prior day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_width = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 1h (each level applies to the entire following day)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for 4h EMA50
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h trend data not ready
        if np.isnan(ema50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + 4h uptrend
            if close[i] > r1_1h[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 + 4h downtrend
            elif close[i] < s1_1h[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price breaks below S1 or 4h trend turns down
            if close[i] < s1_1h[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price breaks above R1 or 4h trend turns up
            if close[i] > r1_1h[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals