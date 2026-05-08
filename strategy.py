# 12h_Camarilla_R4_S4_Breakout_1wTrend_Volume
# Breakout at Camarilla R4/S4 levels with 1w trend filter and volume confirmation
# Uses weekly trend to capture multi-week moves, reducing whipsaw in chop
# R4/S4 levels indicate stronger breakouts, fewer false signals
# Volume surge confirms institutional participation
# Designed for 12h timeframe: targets 50-150 trades over 4 years

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d Camarilla R4/S4 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4 = close_1d + (range_1d * 1.1 / 2)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R4 + weekly uptrend + volume spike
            long_cond = (close[i] > r4_aligned[i]) and \
                        (close[i] > ema_34_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below S4 + weekly downtrend + volume spike
            short_cond = (close[i] < s4_aligned[i]) and \
                         (close[i] < ema_34_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.30
                position = 1
            elif short_cond:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: close below S4 (mean reversion to support)
            if close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: close above R4 (mean reversion to resistance)
            if close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals