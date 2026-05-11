#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend"
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
    open_time = prices['open_time'].values
    
    # Time filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 5 or len(df_1d) < 5:
        return np.zeros(n)
    
    # 4h Camarilla pivot (using previous 4h bar)
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    prev_4h_close = df_4h['close'].shift(1).values
    pivot_4h = (prev_4h_high + prev_4h_low + prev_4h_close) / 3
    r1_4h = 2 * pivot_4h - prev_4h_low
    s1_4h = 2 * pivot_4h - prev_4h_high
    
    # 1d trend filter (EMA34 > EMA89)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_up_1d = ema34_1d > ema89_1d
    trend_down_1d = ema34_1d < ema89_1d
    
    # Align to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in daily uptrend with volume surge
            if (close[i] > r1_4h_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in daily downtrend with volume surge
            elif (close[i] < s1_4h_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 or daily trend changes
            if (close[i] < s1_4h_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R1 or daily trend changes
            if (close[i] > r1_4h_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals