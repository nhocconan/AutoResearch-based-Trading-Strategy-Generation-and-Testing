#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Donchian
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h high/low for Donchian(20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i])
        lower_4h[i] = np.min(low_4h[i-20:i])
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Calculate 4h EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, EMA, volume MA
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.5 * vol_ma
        
        # Entry conditions: breakout with volume and trend filter
        if position == 0:
            # Long: break above upper band + volume + above EMA trend
            if close[i] > upper and vol_breakout and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + below EMA trend
            elif close[i] < lower and vol_breakout and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below EMA or below lower band
            if close[i] < ema_trend or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above EMA or above upper band
            if close[i] > ema_trend or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_EMA34_Trend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0