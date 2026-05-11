#!/usr/bin/env python3
name = "6h_VolumeBreakout_1dTrend_PivotFilter"
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
    
    # 1d trend: EMA34 (weekly is too slow for 6h, use daily trend instead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily pivot levels (classic: PP, R1, S1, R2, S2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume breakout: volume > 2.0 * 50-period SMA of volume (on 6h)
    vol_sma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_breakout = volume > 2.0 * vol_sma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above S1, volume breakout, and above 1d EMA34 (uptrend)
            if close[i] > s1_aligned[i] and vol_breakout[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below R1, volume breakout, and below 1d EMA34 (downtrend)
            elif close[i] < r1_aligned[i] and vol_breakout[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 or below S2 (strong reversal)
            if close[i] < ema_34_1d_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 or above R2 (strong reversal)
            if close[i] > ema_34_1d_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals