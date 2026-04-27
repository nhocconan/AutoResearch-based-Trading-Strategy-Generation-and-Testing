#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day values (shifted by 1)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = high_1d[0]  # first day uses same day
    pl[0] = low_1d[0]
    pc[0] = close_1d[0]
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3.0
    range_ = ph - pl
    
    # Camarilla levels
    r3 = pc + (range_ * 1.1 / 2.0)
    s3 = pc - (range_ * 1.1 / 2.0)
    r4 = pc + (range_ * 1.1)
    s4 = pc - (range_ * 1.1)
    
    # Align levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup period
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > vol_ma[i]
        weekly_trend_up = close[i] > ema_34_1w_aligned[i]
        weekly_trend_down = close[i] < ema_34_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R4 with volume and weekly uptrend
            if close[i] > r4_aligned[i] and vol_ok and weekly_trend_up:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 with volume and weekly downtrend
            elif close[i] < s4_aligned[i] and vol_ok and weekly_trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below R3 or weekly trend turns down
            if close[i] < r3_aligned[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above S3 or weekly trend turns up
            if close[i] > s3_aligned[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0