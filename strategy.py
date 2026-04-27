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
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA50_4h, volume MA, ATR
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_4h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_now = atr_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1d MA
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: trend + volume
        if position == 0:
            # Long: price above EMA50 + volume
            if close[i] > ema_50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below EMA50 + volume
            elif close[i] < ema_50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA50_4hTrend_VolumeFilter_0820UTC"
timeframe = "1h"
leverage = 1.0