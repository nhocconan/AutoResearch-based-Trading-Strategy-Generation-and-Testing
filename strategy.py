#/usr/bin/env python3
name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.083 * range_1d * 1.1 / 2
    s1_1d = close_1d - 1.083 * range_1d * 1.1 / 2
    
    # Align 1d R1/S1 to daily timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (1w EMA50) + volume
            if close[i] > r1_1d_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (1w EMA50) + volume
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the pivot level
            if position == 1:
                if close[i] < s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals