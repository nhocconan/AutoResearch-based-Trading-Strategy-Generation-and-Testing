#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
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
    open_time = prices['open_time'].values
    
    # Hour filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Camarilla pivot (using previous 4h bar)
    ph_4h = df_4h['high'].shift(1).values
    pl_4h = df_4h['low'].shift(1).values
    pc_4h = df_4h['close'].shift(1).values
    p_4h = (ph_4h + pl_4h + pc_4h) / 3
    r1_4h = p_4h + (ph_4h - pl_4h) * 1.1 / 12
    s1_4h = p_4h - (ph_4h - pl_4h) * 1.1 / 12
    
    # 1d trend filter: EMA50 > EMA200
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = ema50_1d > ema200_1d
    trend_down_1d = ema50_1d < ema200_1d
    
    # Align to 1h
    p_4h_aligned = align_htf_to_ltf(prices, df_4h, p_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: current volume > 1.8x 24-period average
    vol_ma24 = np.zeros(n)
    for i in range(n):
        if i < 24:
            vol_ma24[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(p_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1_4h in daily uptrend with volume surge
            if (close[i] > r1_4h_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma24[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1_4h in daily downtrend with volume surge
            elif (close[i] < s1_4h_aligned[i] and 
                  trend_down_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma24[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below pivot or daily trend changes
            if (close[i] < p_4h_aligned[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above pivot or daily trend changes
            if (close[i] > p_4h_aligned[i] or not trend_down_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals