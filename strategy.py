#!/usr/bin/env python3
name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "6h"
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
    
    # 1-week trend: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Camarilla levels (H-L from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    range_1d = high_1d - low_1d
    # Camarilla levels: H/L +/- (range * multiplier)
    r4_1d = close_1d + (range_1d * 1.1000/2)  # R4
    s4_1d = close_1d - (range_1d * 1.1000/2)  # S4
    
    # Align to 6h
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w trend not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R4 AND 1w uptrend AND volume confirmation
            if (close[i] > r4_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S4 AND 1w downtrend AND volume confirmation
            elif (close[i] < s4_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close breaks below S4 OR 1w trend turns down
            if (close[i] < s4_1d_aligned[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close breaks above R4 OR 1w trend turns up
            if (close[i] > r4_1d_aligned[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals