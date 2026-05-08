#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_Volume_Filter"
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
    
    # 1d Keltner Channel (20, 2.0)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema_20 + 2.0 * atr
    lower = ema_20 - 2.0 * atr
    
    # Align to 4h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume spike: current volume > 2.0x 50-period average
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner band, above EMA20, volume spike
            long_cond = (close[i] > upper_aligned[i] and 
                        close[i] > ema_20_aligned[i] and
                        volume_spike[i])
            
            # Short: price breaks below lower Keltner band, below EMA20, volume spike
            short_cond = (close[i] < lower_aligned[i] and 
                         close[i] < ema_20_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA20
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA20
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals