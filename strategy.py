#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 4h data for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_4h = np.zeros(len(high_4h))
    lower_4h = np.zeros(len(low_4h))
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i])
        lower_4h[i] = np.min(low_4h[i-20:i])
    upper_4h[:20] = np.nan
    lower_4h[:20] = np.nan
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get current volume
    volume_now = volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all data ready
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        vol_now = volume_now[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        
        # Volume filter: volume > 1.5x daily MA (volume breakout)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_trend
        trend_down = close[i] < ema_trend
        
        # Entry conditions: breakout with volume and trend alignment
        if position == 0:
            # Long: break above 4h upper band + volume + uptrend
            if close[i] > upper and vol_filter and trend_up:
                signals[i] = size
                position = 1
            # Short: break below 4h lower band + volume + downtrend
            elif close[i] < lower and vol_filter and trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 4h lower band or volume drops
            if close[i] < lower or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 4h upper band or volume drops
            if close[i] > upper or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_EMA34_VolumeBreakout"
timeframe = "4h"
leverage = 1.0