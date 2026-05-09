#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Volume_SessionFilter"
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
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for regime filter (ADX-like)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h volume filter
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_4h_val = ema_4h_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        # Only trade during session
        if not sess_filter:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend alignment: 4h EMA > 1d EMA for uptrend, < for downtrend
        trend_up = ema_4h_val > ema_1d_val
        trend_down = ema_4h_val < ema_1d_val
        
        # Volatility filter: only trade when ATR is elevated (avoid chop)
        vol_condition = atr_1d_val > np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        
        if position == 0:
            # Enter long: 4h above 1d EMA + volume + volatility
            if trend_up and vol_filter and vol_condition:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h below 1d EMA + volume + volatility
            elif trend_down and vol_filter and vol_condition:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend breaks or volume dries up
            if not (trend_up and vol_filter and vol_condition):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend breaks or volume dries up
            if not (trend_down and vol_filter and vol_condition):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals