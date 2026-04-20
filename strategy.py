#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # === Get 1d data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (4h) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any value is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + uptrend + volume
            if (close[i] > dc_upper[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + downtrend + volume
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters Donchian channel or trend reverses
            if close[i] < dc_upper[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Donchian channel or trend reverses
            if close[i] > dc_lower[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals