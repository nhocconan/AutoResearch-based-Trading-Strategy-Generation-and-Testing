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
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(high_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
    donch_upper_12h = upper
    donch_lower_12h = lower
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
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
    start_idx = max(20, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_12h_aligned[i]) or np.isnan(donch_lower_12h_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_12h_aligned[i]
        lower = donch_lower_12h_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.3x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.3 * vol_ma
        
        # Entry conditions: breakout with volume and trend filter
        if position == 0:
            # Long: break above upper band + volume + above weekly EMA
            if close[i] > upper and vol_breakout and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + below weekly EMA
            elif close[i] < lower and vol_breakout and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_WeeklyEMA_VolumeBreakout"
timeframe = "12h"
leverage = 1.0