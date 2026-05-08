#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Volume_Combo"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50 slope
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_slope = np.diff(ema50_4h, prepend=ema50_4h[0])  # slope at each bar
    ema50_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_slope)
    
    # 1d trend filter: price above/below EMA100
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_slope_aligned[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend (EMA50 rising), 1d uptrend (price > EMA100), volume spike
            long_cond = (ema50_4h_slope_aligned[i] > 0 and 
                        close[i] > ema100_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: 4h downtrend (EMA50 falling), 1d downtrend (price < EMA100), volume spike
            short_cond = (ema50_4h_slope_aligned[i] < 0 and 
                         close[i] < ema100_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h downtrend or 1d downtrend
            if ema50_4h_slope_aligned[i] < 0 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h uptrend or 1d uptrend
            if ema50_4h_slope_aligned[i] > 0 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals