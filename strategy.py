#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_EquiVolume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d close for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # EquiVolume: volume-weighted average price over 10 periods
    vwap_num = np.convolve(close * volume, np.ones(10), 'full')[:n]
    vwap_den = np.convolve(volume, np.ones(10), 'full')[:n]
    vwap = np.divide(vwap_num, vwap_den, out=np.zeros_like(vwap_num), where=vwap_den!=0)
    vwap = np.concatenate([np.full(9, np.nan), vwap[9:]])  # align with original index
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need EMA34(34) + VWAP(10)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP and daily uptrend
            long_cond = (close[i] > vwap[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1])
            # Short: price below VWAP and daily downtrend
            short_cond = (close[i] < vwap[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP
            if close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP
            if close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals