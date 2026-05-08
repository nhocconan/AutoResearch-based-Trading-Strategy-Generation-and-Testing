#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pivot_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d volume: 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h Camarilla pivot: H4, L4
    df_1h = get_htf_data(prices, '1h')  # Use 1h for pivot calculation
    if len(df_1h) < 1:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    pivot = (high_1h + low_1h + close_1h) / 3.0
    range_1h = high_1h - low_1h
    h4 = close_1h + (range_1h * 1.1 / 2)
    l4 = close_1h - (range_1h * 1.1 / 2)
    
    h4_aligned = align_htf_to_ltf(prices, df_1h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1h, l4)
    
    # Volume spike: current volume > 2x 1d 20-period average
    volume_spike = volume > (2.0 * vol_ma20_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H4 + uptrend (price > 4h EMA34) + volume spike
            long_cond = (close[i] > h4_aligned[i]) and \
                        (close[i] > ema_34_4h_aligned[i]) and \
                        volume_spike[i]
            # Short: break below L4 + downtrend (price < 4h EMA34) + volume spike
            short_cond = (close[i] < l4_aligned[i]) and \
                         (close[i] < ema_34_4h_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: close below L4 (mean reversion to support)
            if close[i] < l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: close above H4 (mean reversion to resistance)
            if close[i] > h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals