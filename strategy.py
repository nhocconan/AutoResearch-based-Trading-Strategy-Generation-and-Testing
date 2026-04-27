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
    
    # Get 4h data for calculations (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h ATR(14) for volatility
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h_arr, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Volume filter: volume > 1.8x 30-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_4h_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_4h_aligned < atr_median
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i]) or np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA50 + volume filter + low volatility + session
            if (close[i] > ema50_4h_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below EMA50 + volume filter + low volatility + session
            elif (close[i] < ema50_4h_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA50 (trend change)
            if close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above EMA50 (trend change)
            if close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_Vol_LowVol_Session_v1"
timeframe = "1h"
leverage = 1.0