#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_obv_volume_confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for OBV calculation (trend and volume pressure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate OBV (On-Balance Volume) on 1d
    obv_1d = np.zeros(len(close_1d))
    obv_1d[0] = volume_1d[0]
    for i in range(1, len(close_1d)):
        if close_1d[i] > close_1d[i-1]:
            obv_1d[i] = obv_1d[i-1] + volume_1d[i]
        elif close_1d[i] < close_1d[i-1]:
            obv_1d[i] = obv_1d[i-1] - volume_1d[i]
        else:
            obv_1d[i] = obv_1d[i-1]
    
    # Align OBV to 6h timeframe
    obv_1d_aligned = align_htf_to_ltf(prices, df_1d, obv_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA(20) on 1w for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current 6h volume > 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for EMA and OBV
        # Skip if not ready
        if (np.isnan(obv_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # OBV trend: current OBV vs 5-period ago
        if i >= 5:
            obv_prev = obv_1d_aligned[i-5]
            obv_rising = obv_1d_aligned[i] > obv_prev
            obv_falling = obv_1d_aligned[i] < obv_prev
        else:
            obv_rising = False
            obv_falling = False
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry signals: OBV confirmation + volume + trend
        long_signal = obv_rising and volume_ok and uptrend
        short_signal = obv_falling and volume_ok and downtrend
        
        # Exit when OBV reverses direction
        exit_long = obv_falling and position == 1
        exit_short = obv_rising and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals