#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_Volume_Spike_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h data for KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)[:len(change)]
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_12h, np.nan, dtype=np.float64)
    kama[9] = close_12h[9]  # start after 10 periods
    for i in range(10, len(close_12h)):
        if np.isnan(kama[i-1]):
            kama[i] = close_12h[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, price above 1d EMA34, volume spike
            long_cond = (close[i] > kama_aligned[i] and 
                        close[i] > ema34_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: Price below KAMA, price below 1d EMA34, volume spike
            short_cond = (close[i] < kama_aligned[i] and 
                         close[i] < ema34_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below KAMA OR below 1d EMA34
            if close[i] < kama_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above KAMA OR above 1d EMA34
            if close[i] > kama_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals